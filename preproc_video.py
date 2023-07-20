from typing import Dict, Any
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
from pytorch3d.renderer import (MeshRasterizer, MeshRenderer, RasterizationSettings, BlendParams, HardFlatShader, SoftPhongShader, SoftGouraudShader, PointLights, TexturesVertex)
from pytorch3d.structures import Meshes

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


class Renderer:
    def __init__(self, img_size, device):
        self.rast_settings = RasterizationSettings(img_size)
        self.rasterizer = MeshRasterizer(raster_settings=self.rast_settings)
        self.shader = SoftPhongShader(device)
        self.blend_params = BlendParams(background_color=torch.zeros([3], device=device))
        self.mask_shader = HardFlatShader(device, blend_params=self.blend_params)
        self.renderer = MeshRenderer(self.rasterizer, self.shader)
        self.mask_renderer = MeshRenderer(self.rasterizer, self.mask_shader)

        self.point_light = PointLights(
            diffuse_color=torch.tensor([[1.0, 1.0, 1.0]], device=device, dtype=torch.float32), 
            location=torch.tensor([[0.0, 1.0, 1.0]], device=device, dtype=torch.float32), 
            device=device
        )
    
    def render(self, verts, faces, cameras, flame_mask_tex=None):
        
        vcolors = torch.tensor([[0.5, 0.5, 0.5]], device=verts.device)[None].repeat(verts.shape[0], verts.shape[1], 1)
        textures = TexturesVertex(vcolors)
        meshes = Meshes(verts, faces.unsqueeze(0), textures=textures)
        debug_view = self.renderer(meshes, cameras=cameras, lights=self.point_light)

        if flame_mask_tex is not None:
            mask_meshes = Meshes(verts, faces.unsqueeze(0), textures=flame_mask_tex)
            mask = self.mask_renderer(mask_meshes, cameras=cameras)
            return debug_view, mask

        return debug_view, None


class LmkLoss(torch.nn.Module):
    def __init__(
        self, 
        wing_loss_kwargs: Dict[str, Any],
        adaptive_wing_loss_kwargs: Dict[str, Any]=None
    ):
        super(LmkLoss, self).__init__()
        
        self.wing_loss = None
        if wing_loss_kwargs is not None:
            self.wing_loss = nir.WingLoss(**wing_loss_kwargs)
        

        self.adaptive_wing_loss = None
        if adaptive_wing_loss_kwargs is not None:
            self.adaptive_wing_loss = nir.AdaptiveWingLoss(**adaptive_wing_loss_kwargs)
    
    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """"""
        if self.adaptive_wing_loss is not None:
            return self.adaptive_wing_loss(inputs, target)
        return self.wing_loss(inputs, target)


class FLAMEPoseExpressionOptimiation:
    def __init__(
        self, 
        face_parsing_kwargs,
        flame_model_cfg,
        optim_kwargs,
        sched_kwargs,
        loss_kwargs,
        logger_kwargs,
        optim_iters: int=5000,
        cam_init_z_trans: float=-1.0,
        device: str="cuda:0"
    ):
        """"""
        self.device = torch.device(device)
        self.face_parsing = nir.FaceParsing(device=self.device, **face_parsing_kwargs)
        flame_model_cfg = nir.Struct(**flame_model_cfg)
        self.flame_model = nir.FLAME(flame_model_cfg).to(self.device)

        self.optim_kwargs = optim_kwargs
        self.sched_kwargs = sched_kwargs
        self.optim_iters = optim_iters

        self.criterion = LmkLoss(**loss_kwargs)

        self.nshape_params = flame_model_cfg.shape_params
        self.nexpre_params = flame_model_cfg.expression_params

        self.logger = nir.VisdomLogger(**logger_kwargs)

        self.cam_init_z_trans = cam_init_z_trans
    
        self.prev_expression = torch.zeros([1, self.nexpre_params], device=self.device)
        self.prev_pose = torch.zeros([1, 6], device=self.device)
        self.prev_neck_pose = torch.zeros([1, 3], device=self.device)
        self.prev_eye_pose = torch.zeros([1, 6], device=self.device)

        self.prev_camera_trans = torch.tensor([[0.0, 0.0, self.cam_init_z_trans]], device=self.device)
        self.prev_camera_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)

    def reset(self, shapecode: torch.Tensor):
        self.shapecode = shapecode.to(self.device)
    
    def mediapipe_lmks2d_to_screen(self, mplmks2d, width, height):
        mplmks2d[..., 0] = torch.ceil(mplmks2d[..., 0] * height)
        mplmks2d[..., 1] = torch.ceil(mplmks2d[..., 1] * width)
        return mplmks2d.long()
    
    def create_flame_texture(self):
        colormap = self.face_parsing.label_colormap()
        vertex_colors = torch.zeros_like(self.flame_model.v_template)
        leyeball_color = colormap[self.face_parsing.face_segmentor.label_map_11['right_eye']]
        reyeball_color = colormap[self.face_parsing.face_segmentor.label_map_11['left_eye']]
        nose_color = colormap[self.face_parsing.face_segmentor.label_map_11['nose']]
        lips_color = colormap[self.face_parsing.face_segmentor.label_map_11['upper_lip']]
        vertex_colors[self.flame_model.mask_left_eyeball_vidx, :] = torch.from_numpy((leyeball_color / 255.0).astype(np.float32)).to(self.device)
        vertex_colors[self.flame_model.mask_right_eyeball_vidx, :] = torch.from_numpy((reyeball_color / 255.0).astype(np.float32)).to(self.device)
        vertex_colors[self.flame_model.mask_nose_vidx, :] = torch.from_numpy((nose_color / 255.0).astype(np.float32)).to(self.device)
        vertex_colors[self.flame_model.mask_lips_vidx, :] = torch.from_numpy((lips_color / 255.0).astype(np.float32)).to(self.device)
        tex = TexturesVertex(vertex_colors.unsqueeze(0))
        return tex

    def optimization_loop(self, image: torch.Tensor):
        image = torch.from_numpy(image)[None].to(self.device, dtype=torch.float32) / 255.0

        expression_param = torch.nn.Parameter(self.prev_expression.detach(), requires_grad=True)
        pose_param = torch.nn.Parameter(self.prev_pose.detach(), requires_grad=True)
        neck_pose_param = torch.nn.Parameter(self.prev_neck_pose.detach(), requires_grad=True)
        eye_pose_param = torch.nn.Parameter(self.prev_eye_pose.detach(), requires_grad=True)

        camera_trans = torch.nn.Parameter(self.prev_camera_trans.detach(), requires_grad=True)
        camera_quat = torch.nn.Parameter(self.prev_camera_quat, requires_grad=True)

        optim = torch.optim.Adam([expression_param, pose_param, neck_pose_param, eye_pose_param], **self.optim_kwargs)
        sched = torch.optim.lr_scheduler.MultiStepLR(optim, **self.sched_kwargs)

        cam_optim = torch.optim.Adam([camera_trans, camera_quat], **self.optim_kwargs)
        cam_sched = torch.optim.lr_scheduler.MultiStepLR(cam_optim, **self.sched_kwargs)

        aspect_ratio = image.shape[2] / image.shape[1]

        # estimate mediapipe landmarks
        if not self.face_parsing.use_fan:
            mp_flame_corr_idx = self.flame_model.mp_landmark_indices
            mp_lmks_ref = self.face_parsing.parse_lmks((image * 255).to(torch.uint8))[:, mp_flame_corr_idx, 0:2]
            mp_lmks_ref = self.mediapipe_lmks2d_to_screen(mp_lmks_ref, image.shape[1], image.shape[2]).clone().detach()
        else:
            mp_lmks_ref = self.face_parsing.parse_lmks((image * 255).to(torch.uint8))[:, :, 0:2]
    
        # get segmentation mask
        segmentation_mask, lebeled_mask = self.face_parsing.parse_mask((image[0].cpu().numpy() * 255).astype(np.uint8))
        lebeled_mask = torch.from_numpy((lebeled_mask / 255.0).astype(np.float32)).to(self.device)

        flame_mask_texture = self.create_flame_texture()
        flame_renderer = Renderer(image.shape[1:3], self.device)

        for iter in range(self.optim_iters):
            optim.zero_grad()
            cam_optim.zero_grad()

            verts, lmks, mp_lmks = self.flame_model(self.shapecode, expression_param, pose_param, neck_pose_param, eye_pose_param)

            rot = quaternion_to_matrix(camera_quat)
            cameras = FoVPerspectiveCameras(0.01, 1000, 1, R=rot, T=camera_trans).to(self.device)
            lmks2d = cameras.transform_points_screen(lmks, 1e-8, image_size=(image.shape[1], image.shape[2]))[..., 0:2]
            mp_lmks2d = cameras.transform_points_screen(mp_lmks, 1e-8, image_size=(image.shape[1], image.shape[2]))[..., 0:2]

            rendered, rendered_mask = flame_renderer.render(verts, self.flame_model.faces_tensor, cameras, flame_mask_texture)

            mp_lmk_loss = self.criterion(mp_lmks2d, mp_lmks_ref)
            
            segm_loss = torch.abs(rendered_mask[..., 0:3] - lebeled_mask).mean()

            loss = 0.1 * mp_lmk_loss + segm_loss
            
            loss.backward(retain_graph=True)
            optim.step()
            sched.step()
            cam_optim.step()
            cam_sched.step()

            if iter % self.logger.log_iters == 0:
                self.logger.log_msg(f"{iter} | loss: {loss.detach().cpu().item()}")
                self.logger.log_image_w_lmks(image.permute(0, 3, 1, 2), [mp_lmks_ref, mp_lmks2d], 'mediapipe lmks', radius=1)

                self.logger.log_image(rendered_mask[..., 0:3].permute(0, 3, 1, 2), 'rendered mask')
                self.logger.log_image(rendered[..., 0:3].permute(0, 3, 1, 2), 'rendered')
                self.logger.log_image_w_lmks(rendered[..., 0:3].permute(0, 3, 1, 2), mp_lmks2d, 'lmks on flame', radius=1)
                self.logger.log_image(lebeled_mask.permute(0, 3, 1, 2), "face mask")

        self.prev_expression = expression_param.detach()
        self.prev_pose = pose_param.detach()
        self.prev_neck_pose = neck_pose_param.detach()
        self.prev_eye_pose = eye_pose_param.detach()
        self.prev_camera_trans = camera_trans.detach()
        self.prev_camera_quat = camera_quat.detach()



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
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        meshes, shapecode, lmks = mica_estimator.process(image)

        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        flame_optimizer.reset(shapecode)
        flame_optimizer.optimization_loop(image)

        # now optimize for expression and pose
        trimesh.Trimesh(vertices=meshes.cpu() * 1000.0, faces=mica_estimator.faces, process=False).export(os.path.join(output_dir, "mesh.ply"))