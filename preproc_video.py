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

from tqdm import tqdm

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
    """
    A simple pytorch3d renderer for rendering debug flame views
    and flame's semantic masks
    """
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


class OptimizationLoss(torch.nn.Module):
    def __init__(
        self, 
        wing_loss_kwargs: Dict[str, Any],
        w_mp: float=0.2,
        w_seg: float=0.5,
        w_reg: float=1.0,
        adaptive_wing_loss_kwargs: Dict[str, Any]=None
    ):
        super(OptimizationLoss, self).__init__()
        self.w_mp = w_mp
        self.w_seg = w_seg
        self.w_reg = w_reg

        self.wing_loss = None
        if wing_loss_kwargs is not None:
            self.wing_loss = nir.WingLoss(**wing_loss_kwargs)
        
        self.adaptive_wing_loss = None
        if adaptive_wing_loss_kwargs is not None:
            self.adaptive_wing_loss = nir.AdaptiveWingLoss(**adaptive_wing_loss_kwargs)
    
    def expression_reg_loss(self, expression):
        # Normalize the vector to have unit norm
        normalized_vector = torch.nn.functional.normalize(expression, p=2, dim=-1)
        
        # Calculate the mean and subtract it from the normalized vector
        mean_vector = torch.mean(normalized_vector)
        zero_mean_vector = normalized_vector - mean_vector
        
        return zero_mean_vector.mean()

    def forward(
        self, 
        mp_lmks: torch.Tensor, 
        mp_lmks_tgt: torch.Tensor,
        fan_lmks: torch.Tensor,
        fan_lmks_tgt: torch.Tensor,
        seg_mask: torch.Tensor,
        seg_mask_tgt: torch.Tensor,
        expresion_vector: torch.Tensor
    ) -> torch.Tensor:
        """"""
        mp_loss = self.wing_loss(mp_lmks, mp_lmks_tgt)
        fan_loss = self.wing_loss(fan_lmks, fan_lmks_tgt)
        
        seg_mask_loss = torch.abs(seg_mask - seg_mask_tgt).mean()
        # expr_reg = self.expression_reg_loss(expresion_vector)
        output = mp_loss * self.w_mp + fan_loss + seg_mask_loss * self.w_seg + expresion_vector.abs().mean() * self.w_reg
        return output


class FLAMEPoseExpressionOptimization:
    def __init__(
        self, 
        face_parsing_kwargs,
        flame_model_cfg,
        optim_kwargs,
        sched_kwargs,
        loss_kwargs,
        logger_kwargs,
        log_result_only: bool=False,
        optim_iters: int=5000,
        cam_init_z_trans: float=-1.0,
        device: str="cuda:0"
    ):
        """"""
        # configure face_parsing and flame
        self.device = torch.device(device)
        self.face_parsing = nir.FaceParsing(device=self.device, **face_parsing_kwargs)
        flame_model_cfg = nir.Struct(**flame_model_cfg)
        self.flame_model = nir.FLAME(flame_model_cfg).to(self.device)
        self.mp_flame_corr_idx = self.flame_model.mp_landmark_indices

        # keep optimizer state
        self.optim_kwargs = optim_kwargs
        self.sched_kwargs = sched_kwargs
        self.optim_iters = optim_iters

        # configure loss
        self.criterion = OptimizationLoss(**loss_kwargs)

        self.nshape_params = flame_model_cfg.shape_params
        self.nexpre_params = flame_model_cfg.expression_params

        # logger
        self.logger = nir.VisdomLogger(**logger_kwargs)
        self.log_result = log_result_only

        # initialization of optimized parameters
        self.cam_init_z_trans = cam_init_z_trans
    
        self.prev_expression = torch.zeros([1, self.nexpre_params], device=self.device)
        self.prev_global_rot = torch.zeros([1, 3], device=self.device)
        self.prev_jaw_pose = torch.zeros([1, 3], device=self.device)
        self.prev_neck_pose = torch.zeros([1, 3], device=self.device)
        self.prev_eye_pose = torch.zeros([1, 6], device=self.device)

        self.prev_camera_trans = torch.tensor([[0.0, 0.0, self.cam_init_z_trans]], device=self.device)
        self.prev_camera_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)

    def reset(self, shapecode: torch.Tensor, retina_lmks: torch.Tensor):
        self.shapecode = shapecode.to(self.device).detach()
        retina_lmks = retina_lmks[..., 0:2].to(self.device)
        self.retina_lmks = (retina_lmks - retina_lmks.min()) / (retina_lmks.max() - retina_lmks.min())
    
    def lmks2d_to_screen(self, lmks2d, width, height):
        lmks2d[..., 0] = torch.ceil(lmks2d[..., 0] * height)
        lmks2d[..., 1] = torch.ceil(lmks2d[..., 1] * width)
        return lmks2d.long()
    
    def create_flame_mask_texture(self):
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

    def optimization_loop(self, image: torch.Tensor, first_frame: bool=False):
        image = torch.from_numpy(image)[None].to(self.device, dtype=torch.float32) / 255.0

        expression_param = torch.nn.Parameter(self.prev_expression.detach(), requires_grad=True)
        jaw_param = torch.nn.Parameter(self.prev_jaw_pose.detach(), requires_grad=True)
        neck_pose_param = torch.nn.Parameter(self.prev_neck_pose.detach(), requires_grad=True)
        eye_pose_param = torch.nn.Parameter(self.prev_eye_pose.detach(), requires_grad=True)

        camera_trans = torch.nn.Parameter(self.prev_camera_trans.detach(), requires_grad=True)
        camera_quat = torch.nn.Parameter(self.prev_camera_quat, requires_grad=True)
        
        lr = self.optim_kwargs['lr']
        betas = self.optim_kwargs['betas']
        if not first_frame:
            lr = lr * 0.1
        optim = torch.optim.Adam(
            [expression_param, jaw_param, neck_pose_param, eye_pose_param], 
            lr=lr, betas=betas
        )
        sched = torch.optim.lr_scheduler.MultiStepLR(optim, **self.sched_kwargs)

        cam_optim = torch.optim.Adam([camera_trans, camera_quat], lr=lr, betas=betas)
        cam_sched = torch.optim.lr_scheduler.MultiStepLR(cam_optim, **self.sched_kwargs)

        # estimate mediapipe landmarks
        mp_lmks_ref, fan_lmks_ref = self.face_parsing.parse_lmks((image * 255).to(torch.uint8))
        mp_lmks_ref = mp_lmks_ref[:, self.mp_flame_corr_idx, 0:2]
        mp_lmks_ref = self.lmks2d_to_screen(mp_lmks_ref, image.shape[1], image.shape[2]).clone().detach()
        fan_lmks_ref = fan_lmks_ref[..., 0:2].to(self.device)
    
        # get segmentation mask
        segmentation_mask, lebeled_mask = self.face_parsing.parse_mask((image[0].cpu().numpy() * 255).astype(np.uint8))
        lebeled_mask = torch.from_numpy((lebeled_mask / 255.0).astype(np.float32)).to(self.device)

        flame_mask_texture = self.create_flame_mask_texture()
        flame_renderer = Renderer(image.shape[1:3], self.device)

        for iter in tqdm(range(self.optim_iters), total=self.optim_iters, desc=f"progress. init lr: {lr}"):
            optim.zero_grad()
            cam_optim.zero_grad()

            # get shape and landmarks
            pose_param = torch.cat([self.prev_global_rot, jaw_param], dim=-1)
            verts, lmks, mp_lmks = self.flame_model(self.shapecode, expression_param, pose_param, neck_pose_param, eye_pose_param)

            # with the current camera extrinsics
            # transform landmarks to screen
            rot = quaternion_to_matrix(camera_quat)
            cameras = FoVPerspectiveCameras(0.01, 1000, 1, R=rot, T=camera_trans).to(self.device)
            lmks2d = cameras.transform_points_screen(lmks, 1e-8, image_size=(image.shape[1], image.shape[2]))[..., 0:2]
            mp_lmks2d = cameras.transform_points_screen(mp_lmks, 1e-8, image_size=(image.shape[1], image.shape[2]))[..., 0:2]

            # render segmentation mask and debug view
            rendered, rendered_mask = flame_renderer.render(verts, self.flame_model.faces_tensor, cameras, flame_mask_texture)

            # compute los
            loss = self.criterion(mp_lmks2d, mp_lmks_ref, lmks2d, fan_lmks_ref, rendered_mask[..., 0:3], lebeled_mask, expression_param)         
            
            loss.backward(retain_graph=True)
            optim.step()
            sched.step()
            cam_optim.step()
            cam_sched.step()

            if (iter % self.logger.log_iters == 0) and not self.log_result:
                self.logger.log_msg(f"{iter} | loss: {loss.detach().cpu().item()}")
                self.logger.log_image_w_lmks(image.permute(0, 3, 1, 2), [mp_lmks_ref, mp_lmks2d], 'mediapipe lmks', radius=1)
                self.logger.log_image_w_lmks(image.permute(0, 3, 1, 2), [fan_lmks_ref, lmks2d], 'retina lmks', radius=1)

                self.logger.log_image(rendered_mask[..., 0:3].permute(0, 3, 1, 2), 'rendered mask')
                self.logger.log_image(rendered[..., 0:3].permute(0, 3, 1, 2), 'rendered')
                self.logger.log_image_w_lmks(rendered[..., 0:3].permute(0, 3, 1, 2), mp_lmks2d, 'lmks on flame', radius=1)
                self.logger.log_image(lebeled_mask.permute(0, 3, 1, 2), "face mask")
        
        if self.log_result:
            self.logger.log_image_w_lmks(image.permute(0, 3, 1, 2), [mp_lmks_ref, mp_lmks2d], 'mediapipe lmks', radius=1)
            self.logger.log_image_w_lmks(image.permute(0, 3, 1, 2), [fan_lmks_ref, lmks2d], 'retina lmks', radius=1)

            self.logger.log_image(rendered_mask[..., 0:3].permute(0, 3, 1, 2), 'rendered mask')
            self.logger.log_image(rendered[..., 0:3].permute(0, 3, 1, 2), 'rendered')
            self.logger.log_image_w_lmks(rendered[..., 0:3].permute(0, 3, 1, 2), mp_lmks2d, 'lmks on flame', radius=1)
            self.logger.log_image(lebeled_mask.permute(0, 3, 1, 2), "face mask")

        self.prev_expression = expression_param.detach()
        self.prev_global_rot = pose_param[:, 0:3].detach()
        self.prev_jaw_pose = pose_param[:, 3:].detach()
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
    flame_optimizer = FLAMEPoseExpressionOptimization(**conf.flame_pose_expression_optimization_kwargs)

    # create dataset
    dataset = nir.get_dataset("SingleVideoDataset", **conf.video_dataset_kwargs)


    for frame_idx in tqdm(enumerate(dataset), total=len(dataset), desc="optimizing frame"):
        data = dataset.__getitem__(frame_idx)
        # estimator needs numpy array
        image = (data.rgb.cpu().numpy() * 255).astype(np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        meshes, shapecode, lmks = mica_estimator.process(image)

        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        flame_optimizer.reset(shapecode, lmks)
        flame_optimizer.optimization_loop(image, True if frame_idx == 80 else False)

        # now optimize for expression and pose
        trimesh.Trimesh(vertices=meshes.cpu() * 1000.0, faces=mica_estimator.faces, process=False).export(os.path.join(output_dir, "mesh.ply"))