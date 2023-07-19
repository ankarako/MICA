import argparse
import os
import random
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import trimesh
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger
from skimage.io import imread
from tqdm import tqdm

from configs.config import get_cfg_defaults
from datasets.creation.util import get_arcface_input, get_center, draw_on
from utils import util
from utils.landmark_detector import LandmarksDetector, detectors

import yaml
import nir

from pytorch3d.renderer.cameras import FoVPerspectiveCameras
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion


def deterministic(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    np.random.seed(rank)
    random.seed(rank)

    cudnn.deterministic = True
    cudnn.benchmark = False


def load_checkpoint(args, mica):
    checkpoint = torch.load(args.m)
    if 'arcface' in checkpoint:
        mica.arcface.load_state_dict(checkpoint['arcface'])
    if 'flameModel' in checkpoint:
        mica.flameModel.load_state_dict(checkpoint['flameModel'])

class FLAMEPoseExpressionOptimiation:
    class Loss:
        def __init__(self, w):
            self.w = w

        def __call__(self, lmks, lmks_ref):
            lmk_loss = torch.nn.functional.l1_loss(lmks, lmks_ref)
            return lmk_loss
        
    def __init__(
        self, 
        face_parsing_kwargs,
        flame_model_cfg,
        optim_kwargs,
        sched_kwargs,
        loss_kwargs,
        logger_kwargs,
        optim_iters: int=1000,
        device: str="cuda:0"
    ):
        """"""
        self.device = torch.device(device)
        self.face_parsing = nir.FaceParsing(**face_parsing_kwargs)
        flame_model_cfg = nir.Struct(**flame_model_cfg)
        self.flame_model = nir.FLAME(flame_model_cfg).to(self.device)

        self.optim_kwargs = optim_kwargs
        self.sched_kwargs = sched_kwargs
        self.optim_iters = optim_iters

        self.criterion = self.Loss(**loss_kwargs)

        self.nshape_params = flame_model_cfg.shape_params
        self.nexpre_params = flame_model_cfg.expression_params

        self.logger = nir.TensorboardLogger(**logger_kwargs)

    def reset(self, shapecode: torch.Tensor):
        self.shapecode = shapecode.to(self.device)
    

    def mediapipe_lmks2d_to_screen(self, mplmks3d, width, height):
        output = mplmks3d.clone()
        output = output[..., 0:2]
        output[..., 0] = torch.ceil(output[..., 0] * height)
        output[..., 1] = torch.ceil(output[..., 1] * width)
        return output.long()

    def optimization_loop(self, image: torch.Tensor):
        image = torch.from_numpy(image)[None].to(self.device)

        expression_param = torch.nn.Parameter(torch.zeros([1, self.nexpre_params], device=self.device), requires_grad=True)
        pose_param = torch.nn.Parameter(torch.zeros([1, 6], device=self.device), requires_grad=True)
        neck_pose_param = torch.nn.Parameter(torch.zeros([1, 3], device=self.device), requires_grad=True)
        eye_pose_param = torch.nn.Parameter(torch.zeros([1, 6], device=self.device), requires_grad=True)

        camera_trans = torch.nn.Parameter(torch.tensor([[0.0, 0.0, -1.5]], device=self.device), requires_grad=True)
        camera_quat = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device), requires_grad=True)

        optim = torch.optim.Adam([expression_param, pose_param, neck_pose_param, eye_pose_param], **self.optim_kwargs)
        sched = torch.optim.lr_scheduler.MultiStepLR(optim, **self.sched_kwargs)

        cam_optim = torch.optim.Adam([camera_trans, camera_quat], **self.optim_kwargs)
        cam_sched = torch.optim.lr_scheduler.MultiStepLR(cam_optim, **self.sched_kwargs)

        aspect_ratio = image.shape[1] / image.shape[2]

        # estimate mediapipe landmarks
        mp_flame_corr_idx = self.flame_model.mp_landmark_indices
        mp_lmks_ref = self.face_parsing.parse_lmks(image)[:, mp_flame_corr_idx, :]
        mp_lmks_ref = self.mediapipe_lmks2d_to_screen(mp_lmks_ref, image.shape[0], image.shape[1]).clone().detach()
    

        for iter in range(self.optim_iters):
            optim.zero_grad()
            cam_optim.zero_grad()

            verts, lmks, mp_lmks = self.flame_model(self.shapecode, expression_param, pose_param, neck_pose_param, eye_pose_param)

            rot = quaternion_to_matrix(camera_quat)
            cameras = FoVPerspectiveCameras(0.01, 1000, aspect_ratio, R=rot, T=camera_trans).to(self.device)
            mp_lmks2d = cameras.transform_points_screen(mp_lmks, 1e-8, image_size=(image.shape[1], image.shape[2]))[..., 0:2]

            loss = self.criterion(mp_lmks2d, mp_lmks_ref)

            loss.backward(retain_graph=True)
            optim.step()
            sched.step()
            cam_optim.step()
            cam_sched.step()

            if iter % self.logger.log_iters == 0:
                self.logger.log_msg(f"{iter} | loss: {loss.detach().cpu().item()}")
                self.logger.log_image_w_lmks(image, mp_lmks_ref, 'mediapipe lmks', iter, "HWC")
                self.logger.log_image_w_lmks(image, mp_lmks2d, 'estimated lmks', iter, "HWC")




class MicaEstimator:
    def __init__(
        self,
        chkp_path: str,
        device: str="cuda:0"
    ):
        self.cfg = get_cfg_defaults()
        self.device = torch.device(device)

        # create MICA
        self.mica = util.find_model_using_name(
            model_dir='micalib.models', model_name=self.cfg.model.name)(self.cfg, self.device)
        self.mica.testing = True
        # load MICA checkpoin
        assert os.path.exists(chkp_path), "The specified checkpoint path does not exist"
        checkpoint = torch.load(chkp_path)
        if 'arcface' in checkpoint:
            self.mica.arcface.load_state_dict(checkpoint['arcface'])
        if 'flameModel' in checkpoint:
            self.mica.flameModel.load_state_dict(checkpoint['flameModel'])
        
        self.mica.eval()
        self.faces = self.mica.flameModel.generator.faces_tensor.cpu()
        self.app = LandmarksDetector(model=detectors.RETINAFACE)
    
    def process(self, image: np.ndarray, image_size: int=224):
        # detect keypoints and crop
        bboxes, kpss = self.app.detect(image)
        if bboxes.shape[0] == 0:
            return None
        
        i = get_center(bboxes, image)
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i]
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        blob, aimg = get_arcface_input(face, image)
        cropped_image = face_align.norm_crop(image, landmark=face.kps, image_size=image_size)

        # run MICA
        cropped_image = (cropped_image / 255.0).astype(np.float32)
        cropped_image = torch.from_numpy(cropped_image).permute(2, 0, 1).cuda()[None]

        blob = torch.from_numpy(blob).cuda()[None]
        codedict = self.mica.encode(cropped_image, blob)
        opdict = self.mica.decode(codedict)

        meshes = opdict["pred_canonical_shape_vertices"]
        code = opdict["pred_shape_code"]
        lmk = self.mica.flame.compute_landmarks(meshes)
        return meshes[0].detach().cpu(), code, lmk


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MICA video estimator")
    parser.add_argument("--conf", type=str, help="The path to the configuration file")
    args = parser.parse_args()

    deterministic(42)

    # create output directory
    assert os.path.exists(args.conf)
    with open(args.conf, 'r') as infd:
        conf = yaml.safe_load(infd)
    
    conf = nir.Struct(**conf)

    output_dir = os.path.abspath(os.path.join(os.path.dirname(conf.input_video), os.path.basename(conf.input_video).replace(".mp4", ""))) if conf.output is None else conf.output
    if not os.path.exists(output_dir):
        os.mkdir(os.path.join(output_dir))

    # create the estimators
    mica_estimator = MicaEstimator(**conf.mica_estimator_kwargs)
    flame_optimizer = FLAMEPoseExpressionOptimiation(**conf.flame_pose_expression_optimization_kwargs)

    # create dataset
    dataset = nir.get_dataset("SingleVideoDataset", **conf.video_dataset_kwargs)

    for frame_idx, data in enumerate(dataset):
        # estimator needs numpy array
        image = (data.rgb.cpu().numpy() * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        meshes, shapecode, lmks = mica_estimator.process(image)

        flame_optimizer.reset(shapecode)
        flame_optimizer.optimization_loop(image)

        # now optimize for expression and pose
        trimesh.Trimesh(vertices=meshes.cpu() * 1000.0, faces=mica_estimator.faces, process=False).export(os.path.join(output_dir, "mesh.ply"))