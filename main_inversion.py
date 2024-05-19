import os
import warnings

warnings.filterwarnings("ignore")

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print(rlimit)
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, rlimit[1]))
from itertools import chain

import hydra
from hydra.utils import instantiate

from typing import Optional
from omegaconf import DictConfig, OmegaConf
from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from torchmetrics.functional.image import image_gradients
from typing import Any, Callable, Dict, Optional, Tuple, List
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything, Trainer, LightningModule

from pytorch3d.renderer.cameras import (
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    look_at_view_transform,
)
from pytorch3d.renderer.camera_utils import join_cameras_as_batch

from omegaconf import OmegaConf
from PIL import Image
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.implicitron.dataset.utils import DATASET_TYPE_KNOWN, DATASET_TYPE_UNKNOWN
from pytorch3d.implicitron.dataset.rendered_mesh_dataset_map_provider import (
    RenderedMeshDatasetMapProvider,
)

from pytorch3d.implicitron.models.generic_model import GenericModel
from pytorch3d.implicitron.models.implicit_function.base import (
    ImplicitFunctionBase,
    ImplicitronRayBundle,
)
from pytorch3d.implicitron.models.renderer.raymarcher import (
    AccumulativeRaymarcherBase,
    RaymarcherBase,
)
from pytorch3d.implicitron.models.renderer.base import (
    BaseRenderer,
    RendererOutput,
    EvaluationMode,
    ImplicitFunctionWrapper,
)
from pytorch3d.implicitron.models.renderer.multipass_ea import (
    MultiPassEmissionAbsorptionRenderer,
)
from pytorch3d.implicitron.models.renderer.ray_point_refiner import RayPointRefiner
from pytorch3d.implicitron.tools.config import (
    get_default_args,
    registry,
    remove_unused_components,
    run_auto_creation,
)
from pytorch3d.renderer.implicit.renderer import VolumeSampler
from pytorch3d.vis.plotly_vis import plot_batch_individually, plot_scene
from pytorch3d.renderer.implicit.raymarching import (
    _check_density_bounds,
    _check_raymarcher_inputs,
    _shifted_cumprod,
)

from monai.losses import PerceptualLoss
from monai.networks.nets import UNet
from monai.networks.layers.factories import Norm

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

from datamodule import UnpairedDataModule
from dvr.renderer import ReverseXRayVolumeRenderer
from dvr.renderer import normalized
from dvr.renderer import standardized


def make_cameras_dea(
    dist: torch.Tensor,
    elev: torch.Tensor,
    azim: torch.Tensor,
    fov: int = 40,
    znear: int = 4.0,
    zfar: int = 8.0,
    is_orthogonal: bool = False,
):
    assert dist.device == elev.device == azim.device
    _device = dist.device
    R, T = look_at_view_transform(dist=dist, elev=elev * 90, azim=azim * 180)
    if is_orthogonal:
        return FoVOrthographicCameras(R=R, T=T, znear=znear, zfar=zfar).to(_device)
    return FoVPerspectiveCameras(R=R, T=T, fov=fov, znear=znear, zfar=zfar).to(_device)


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class NVLightningModule(LightningModule):
    def __init__(self, model_cfg: DictConfig, train_cfg: DictConfig):
        super().__init__()

        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        # print(self.model_cfg.img_shape, self.model_cfg.vol_shape, self.model_cfg.fov_depth)
        self.fwd_renderer = ReverseXRayVolumeRenderer(
            image_width=self.model_cfg.img_shape,
            image_height=self.model_cfg.img_shape,
            n_pts_per_ray=self.model_cfg.n_pts_per_ray,
            min_depth=self.model_cfg.min_depth,
            max_depth=self.model_cfg.max_depth,
            ndc_extent=self.model_cfg.ndc_extent,
        ).float()

        # @ Diffusion 
        self.unet3d_model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=self.model_cfg.fov_depth+self.model_cfg.vol_shape+self.model_cfg.fov_depth,
            num_channels=[256, 256, 512],
            attention_levels=[False, False, True],
            num_head_channels=[0, 0, 512],
            num_res_blocks=2,
            with_conditioning=True, 
            cross_attention_dim=12, # Condition with straight/hidden view  # flatR | flatT
        )
        init_weights(self.unet3d_model, init_type="normal")

        self.vnet3d_model = UNet(
            spatial_dims=3, 
            in_channels=3, 
            out_channels=1, 
            channels=(32, 64, 128, 256, 512), #(32, 64, 128, 256, 512), #(32, 48, 80, 224, 640), 
            strides=(2, 2, 2, 2), 
            num_res_units=2, 
            kernel_size=3, 
            up_kernel_size=3, 
            act=("LeakyReLU", {"inplace": True}), 
            norm=Norm.BATCH,
            # dropout=0.5
        )

        if self.train_cfg.perceptual:
            self.p2dloss = PerceptualLoss(
                spatial_dims=2, 
                network_type="radimagenet_resnet50", 
                is_fake_3d=False, 
                pretrained=True,
            ).float()
            self.p3dloss = PerceptualLoss(
                spatial_dims=3, 
                network_type="medicalnet_resnet50_23datasets", 
                is_fake_3d=False, 
                pretrained=True,
            ).float()

        # @ Diffusion 
        self.unet2d_model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=[256, 256, 512],
            attention_levels=[False, False, True],
            num_head_channels=[0, 0, 512],
            num_res_blocks=2,
            with_conditioning=True, 
            cross_attention_dim=12, # Condition with straight/hidden view  # flatR | flatT
        )
        init_weights(self.unet2d_model, init_type="normal")

        self.ddpmsch = DDPMScheduler(
            num_train_timesteps=self.model_cfg.timesteps, 
            prediction_type=self.model_cfg.prediction_type, 
            schedule="scaled_linear_beta", 
            beta_start=0.0005, 
            beta_end=0.0195,
        )

        self.ddimsch = DDIMScheduler(
            num_train_timesteps=self.model_cfg.timesteps, 
            prediction_type=self.model_cfg.prediction_type, 
            schedule="scaled_linear_beta", 
            beta_start=0.0005, 
            beta_end=0.0195, 
        )

        self.inferer = DiffusionInferer(
            scheduler=self.ddpmsch
        )

        if self.model_cfg.phase=="finetune":
            pass
        
        if self.train_cfg.ckpt:
            print("Loading.. ", self.train_cfg.ckpt)
            checkpoint = torch.load(self.train_cfg.ckpt, map_location=torch.device("cpu"))["state_dict"]
            state_dict = {k: v for k, v in checkpoint.items() if k in self.state_dict()}
            self.load_state_dict(state_dict, strict=False)

        self.save_hyperparameters()
        self.train_step_outputs = []
        self.validation_step_outputs = []

    def correct_window(self, T_old, a_min=-1024, a_max=3071, b_min=-512, b_max=3071):
        # Calculate the range for the old and new scales
        range_old = a_max - a_min
        range_new = b_max - b_min

        # Reverse the incorrect scaling
        T_raw = (T_old * range_old) + a_min

        # Apply the correct scaling
        T_new = (T_raw - b_min) / range_new
        return T_new.clamp_(0, 1)

    def forward_screen(self, image3d, cameras, is_training=False):
        image3d = self.correct_window(image3d, a_min=-1024, a_max=3071, b_min=-512, b_max=3071)
        image2d = self.fwd_renderer(image3d.float(), cameras, norm_type="standardized", stratified_sampling=is_training)
        return image2d
   
    def flatten_cameras(self, cameras, zero_translation=True):
        camera_ = cameras.clone()
        R = camera_.R
        if zero_translation:
            T = torch.zeros_like(camera_.T.unsqueeze_(-1))
        else:
            T = camera_.T.unsqueeze_(-1)
        return torch.cat([R.reshape(-1, 1, 9), T.reshape(-1, 1, 3)], dim=-1).contiguous().view(-1, 1, 12)

    def forward_volume(self, image2d, cameras, noise=None, timesteps=None):
        _device = image2d.device
        B = image2d.shape[0]
        timesteps = torch.zeros((B,), device=_device).long() if timesteps is None else timesteps

        mat = self.flatten_cameras(cameras, zero_translation=True)

        image2d = torch.rot90(image2d, 1, [2, 3])
        image2d = torch.flip(image2d, [3])
        results = self.inferer(
            inputs=image2d, 
            noise=noise, 
            diffusion_model=self.unet3d_model, 
            condition=mat.view(-1, 1, 12), 
            timesteps=timesteps
        ) 

        fov = results[:, :self.model_cfg.fov_depth,...].unsqueeze(1)
        vol = results[:, -self.model_cfg.vol_shape:,...].unsqueeze(1)
        oth = results[:, self.model_cfg.fov_depth:-self.model_cfg.vol_shape,...].unsqueeze(1)
        
        # Let us resample the field of view 
        cam = cameras.clone()
        R = cam.R
        # T = torch.zeros_like(cam.T)

        z = torch.linspace(-1.0, 1.0, steps=self.model_cfg.vol_shape, device=_device)
        y = torch.linspace(-1.0, 1.0, steps=self.model_cfg.vol_shape, device=_device)
        x = torch.linspace(-1.0, 1.0, steps=self.model_cfg.vol_shape, device=_device)
        grd_fov = torch.stack(torch.meshgrid(x, y, z), dim=-1).view(-1, 3).unsqueeze(0).repeat(B, 1, 1)  # 1 DHW 3 to B DHW 3
        # Process (resample) the volumes from ray views to ndc
        pts_fov = cam.transform_points_ndc(grd_fov)  # world to ndc, 1 DHW 3
        out_fov = F.grid_sample(
            fov.float(), 
            pts_fov.view(-1, self.model_cfg.vol_shape, self.model_cfg.vol_shape, self.model_cfg.vol_shape, 3).float(), 
            mode="bilinear", 
            padding_mode="border", 
            align_corners=True,
        )
        out_fov = torch.permute(out_fov, [0, 1, 4, 3, 2]).nan_to_num_().float()
        
        # Let us resample the volume
        cam = cameras.clone()
        R = cam.R
        T = cam.T.unsqueeze_(-1)
        # T = torch.zeros_like(cam.T.unsqueeze_(-1))
        
        mat = torch.cat([R, T], dim=-1)
        inv = torch.cat([torch.inverse(R), -T], dim=-1)

        grd_oth = F.affine_grid(inv, out_fov.size()).type(image2d.dtype)
        out_oth = F.grid_sample(oth, grd_oth)

        results = self.vnet3d_model(torch.cat([out_fov, vol, out_oth], dim=1))
        return results
    
    def forward_timing(self, image2d, cameras, noise=None, timesteps=None):
        _device = image2d.device
        B = image2d.shape[0]
        timesteps = torch.zeros((B,), device=_device).long() if timesteps is None else timesteps

        mat = self.flatten_cameras(cameras, zero_translation=True)

        results = self.inferer(
            inputs=image2d, 
            noise=noise, 
            diffusion_model=self.unet2d_model, 
            condition=mat.view(-1, 1, 12), 
            timesteps=timesteps
        ) 
        return results
    
    def _common_step(self, batch, batch_idx, stage: Optional[str] = "evaluation"):
        image2d = batch["image2d"]
        image3d = batch["image3d"]
        _device = batch["image3d"].device
        B = image2d.shape[0]

        # self.ddimsch.set_timesteps(num_train_timesteps=self.model_cfg.timesteps)
        # Construct the random cameras, -1 and 1 are the same point in azimuths
        dist_random = 8 * torch.ones(B, device=_device)
        elev_random = torch.rand_like(dist_random) - 0.5
        azim_random = torch.rand_like(dist_random) * 2 - 1  # from [0 1) to [-1 1)
        view_random = make_cameras_dea(dist_random, elev_random, azim_random, fov=self.model_cfg.fov, znear=self.model_cfg.min_depth, zfar=self.model_cfg.max_depth)

        dist_second = 8 * torch.ones(B, device=_device)
        elev_second = torch.rand_like(dist_second) - 0.5
        azim_second = torch.rand_like(dist_second) * 2 - 1  # from [0 1) to [-1 1)
        view_second = make_cameras_dea(dist_second, elev_second, azim_second, fov=self.model_cfg.fov, znear=self.model_cfg.min_depth, zfar=self.model_cfg.max_depth)

        dist_hidden = 8 * torch.ones(B, device=_device)
        elev_hidden = torch.zeros(B, device=_device)
        azim_hidden = torch.zeros(B, device=_device)
        view_hidden = make_cameras_dea(dist_hidden, elev_hidden, azim_hidden, fov=self.model_cfg.fov, znear=self.model_cfg.min_depth, zfar=self.model_cfg.max_depth)

        # Construct the samples in 2D
        figure_xr_source_hidden = image2d
        figure_ct_source_random = self.forward_screen(image3d=image3d, cameras=view_random)
        figure_ct_source_second = self.forward_screen(image3d=image3d, cameras=view_second)

        timesteps = torch.randint(0, self.ddimsch.num_train_timesteps, (B,), device=_device).long()  
            
        volume_xr_latent = torch.empty_like(image3d)
        volume_ct_latent = torch.empty_like(image3d)
        nn.init.trunc_normal_(volume_xr_latent, mean=0.50, std=0.25, a=0, b=1)
        nn.init.trunc_normal_(volume_ct_latent, mean=0.50, std=0.25, a=0, b=1)
        
        figure_xr_latent_hidden = self.forward_screen(image3d=volume_xr_latent, cameras=view_hidden)
        figure_ct_latent_random = self.forward_screen(image3d=volume_ct_latent, cameras=view_random)
        figure_ct_latent_second = self.forward_screen(image3d=volume_ct_latent, cameras=view_second)

        # Run the forward pass
        figure_dx_source_concat = torch.cat([figure_xr_source_hidden, figure_ct_source_random, figure_ct_source_second])
        figure_dx_latent_concat = torch.cat([figure_xr_latent_hidden, figure_ct_latent_random, figure_ct_latent_second])
        volume_dx_latent_concat = torch.cat([volume_xr_latent, volume_ct_latent, volume_ct_latent])
        camera_dx_render_concat = join_cameras_as_batch([view_hidden, view_random, view_second])

        # For 2D
        figure_dx_output_concat = self.forward_timing(
            image2d=figure_dx_source_concat, 
            cameras=camera_dx_render_concat, 
            noise=figure_dx_latent_concat, 
            timesteps=timesteps.repeat(3),
        )

        # Split the tensor
        figure_xr_output_hidden, \
        figure_ct_output_random, \
        figure_ct_output_second = torch.split(figure_dx_output_concat, B, dim=0)

        # Set the target
        if self.ddpmsch.prediction_type == "sample":
            figure_xr_target_hidden = figure_xr_source_hidden
            figure_ct_target_random = figure_ct_source_random
            figure_ct_target_second = figure_ct_source_second
        elif self.ddpmsch.prediction_type == "epsilon":
            figure_xr_target_hidden = figure_xr_latent_hidden
            figure_ct_target_random = figure_ct_latent_random
            figure_ct_target_second = figure_ct_latent_second
        elif self.ddpmsch.prediction_type == "v_prediction":
            figure_xr_target_hidden = self.ddpmsch.get_velocity(figure_xr_source_hidden, figure_xr_latent_hidden, timesteps)
            figure_ct_target_random = self.ddpmsch.get_velocity(figure_ct_source_random, figure_ct_latent_random, timesteps)
            figure_ct_target_second = self.ddpmsch.get_velocity(figure_ct_source_second, figure_ct_latent_second, timesteps)
        
        im2d_loss_dif = F.l1_loss(figure_xr_output_hidden, figure_xr_target_hidden) \
                      + F.l1_loss(figure_ct_output_random, figure_ct_target_random) \
                      + F.l1_loss(figure_ct_output_second, figure_ct_target_second) 
        
        # For 3D
        volume_dx_output_concat = self.forward_volume(
            image2d=figure_dx_output_concat, 
            cameras=camera_dx_render_concat, 
            noise=torch.zeros_like(figure_dx_latent_concat), 
            timesteps=timesteps.repeat(3),
        )

         # Split the tensor
        volume_xr_output_hidden, \
        volume_ct_output_random, \
        volume_ct_output_second = torch.split(volume_dx_output_concat, B, dim=0)

        # Set the target
        if self.ddpmsch.prediction_type == "sample":
            volume_ct_target_random = image3d
            volume_ct_target_second = image3d
        elif self.ddpmsch.prediction_type == "epsilon":
            volume_ct_target_random = volume_ct_latent
            volume_ct_target_second = volume_ct_latent
        elif self.ddpmsch.prediction_type == "v_prediction":
            volume_ct_target_random = self.ddpmsch.get_velocity(image3d, volume_ct_latent, timesteps)
            volume_ct_target_second = self.ddpmsch.get_velocity(image3d, volume_ct_latent, timesteps)
        
        im3d_loss_dif = F.l1_loss(volume_ct_output_random, volume_ct_target_random) \
                      + F.l1_loss(volume_ct_output_second, volume_ct_target_second) 
        
        if self.train_cfg.perceptual:
            figure_xr_output_hidden_hidden = self.forward_screen(image3d=volume_xr_output_hidden, cameras=view_hidden)
            figure_xr_output_hidden_random = self.forward_screen(image3d=volume_xr_output_hidden, cameras=view_random)
        figure_ct_output_random_random = self.forward_screen(image3d=volume_ct_output_random, cameras=view_random)
        figure_ct_output_second_second = self.forward_screen(image3d=volume_ct_output_second, cameras=view_second)
        figure_ct_output_random_second = self.forward_screen(image3d=volume_ct_output_random, cameras=view_second)
        figure_ct_output_second_random = self.forward_screen(image3d=volume_ct_output_second, cameras=view_random)

        im2d_loss_cyc = F.l1_loss(figure_ct_output_random_random, figure_ct_target_random) \
                      + F.l1_loss(figure_ct_output_second_second, figure_ct_target_second) \
                      + F.l1_loss(figure_ct_output_random_second, figure_ct_target_second) \
                      + F.l1_loss(figure_ct_output_second_random, figure_ct_target_random) \
        
        im2d_loss = im2d_loss_dif + im2d_loss_cyc
        im3d_loss = im3d_loss_dif
        loss = self.train_cfg.alpha * im2d_loss + self.train_cfg.gamma * im3d_loss
        
        if self.train_cfg.perceptual and stage=="train":
            pc2d_loss = self.p2dloss(figure_xr_output_hidden_random.float(), figure_ct_source_random.float()) \
                      + self.p2dloss(figure_xr_output_hidden_hidden.float(), figure_xr_target_hidden.float()) 
            loss += self.train_cfg.lamda * pc2d_loss    
            pc3d_loss = self.p3dloss(volume_xr_output_hidden.float(), image3d.float()) 
            loss += self.train_cfg.lamda * pc3d_loss

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=B)
        
        # Visualization step
        if batch_idx == 0:
            # Sampling step for X-ray
            with torch.no_grad():
                ###
                figure_dx_sample_concat = torch.empty_like(figure_dx_latent_concat)
                nn.init.trunc_normal_(figure_dx_sample_concat, mean=0.50, std=0.25, a=0, b=1)
                mat = self.flatten_cameras(camera_dx_render_concat, zero_translation=True)
                self.ddimsch.set_timesteps(num_inference_steps=self.model_cfg.timesteps//10)
                figure_dx_sample_concat = self.inferer.sample(
                    input_noise=figure_dx_sample_concat, 
                    conditioning=mat.view(-1, 1, 12), 
                    diffusion_model=self.unet2d_model, 
                    scheduler=self.ddimsch,
                    verbose=False,
                )

                figure_xr_sample_hidden, \
                figure_ct_sample_random, \
                figure_ct_sample_second = torch.split(figure_dx_sample_concat, B, dim=0)

                ###
                figure_dx_remake_concat = figure_dx_source_concat
                volume_dx_remake_concat = self.forward_volume(
                    image2d=figure_dx_remake_concat, 
                    cameras=camera_dx_render_concat, 
                    noise=torch.zeros_like(figure_dx_remake_concat),
                    timesteps=None,
                )

                volume_xr_remake_hidden, \
                volume_ct_remake_random, \
                volume_ct_remake_second = torch.split(volume_dx_remake_concat, B, dim=0)

                figure_xr_remake_hidden_hidden = self.forward_screen(image3d=volume_xr_remake_hidden, cameras=view_hidden)
                figure_xr_remake_hidden_random = self.forward_screen(image3d=volume_xr_remake_hidden, cameras=view_random)
                figure_ct_remake_random_random = self.forward_screen(image3d=volume_ct_remake_random, cameras=view_random)
                figure_ct_remake_second_second = self.forward_screen(image3d=volume_ct_remake_second, cameras=view_second)
                figure_ct_remake_random_second = self.forward_screen(image3d=volume_ct_remake_random, cameras=view_second)
                figure_ct_remake_second_random = self.forward_screen(image3d=volume_ct_remake_second, cameras=view_random)

                zeros = torch.zeros_like(image2d)
                ###
                viz2d = torch.cat([
                    torch.cat([
                        image2d, 
                        volume_xr_remake_hidden[..., self.model_cfg.vol_shape // 2, :], 
                        figure_xr_remake_hidden_hidden,
                        figure_xr_remake_hidden_random, 
                    # ], dim=-2).transpose(2, 3),
                    # torch.cat([
                        figure_ct_source_random, 
                        volume_ct_remake_random[..., self.model_cfg.vol_shape // 2, :], 
                        figure_ct_remake_random_random,
                        figure_ct_remake_random_second, 
                    ], dim=-2).transpose(2, 3),
                    torch.cat([
                        figure_xr_sample_hidden, 
                        image3d[..., self.model_cfg.vol_shape // 2, :], 
                        figure_ct_sample_random,
                        figure_ct_sample_second, 
                    # ], dim=-2).transpose(2, 3),
                    # torch.cat([
                        figure_ct_source_second, 
                        volume_ct_remake_second[..., self.model_cfg.vol_shape // 2, :], 
                        figure_ct_remake_second_random,
                        figure_ct_remake_second_second, 
                    ], dim=-2).transpose(2, 3),
                ], dim=-2)

                tensorboard = self.logger.experiment
                grid2d = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=1, padding=0).clamp(0, 1)
                tensorboard.add_image(f"{stage}_df_samples", grid2d, self.current_epoch * B + batch_idx)    
        return loss
                        
    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, stage="train")
        self.train_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, stage="validation")
        self.validation_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        loss = torch.stack(self.train_step_outputs).mean()
        self.log(
            f"train_loss_epoch",
            loss,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.train_step_outputs.clear()  # free memory

    def on_validation_epoch_end(self):
        loss = torch.stack(self.validation_step_outputs).mean()
        self.log(
            f"validation_loss_epoch",
            loss,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.validation_step_outputs.clear()  # free memory

    def sample(self, **kwargs: dict):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.train_cfg.lr, betas=(0.5, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 200], gamma=0.1
        )
        return [optimizer], [scheduler]


@hydra.main(version_base=None, config_path="./conf")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)  # resolve all str interpolation
    seed_everything(42)
    datamodule = UnpairedDataModule(
        train_image3d_folders=cfg.data.train_image3d_folders,
        train_image2d_folders=cfg.data.train_image2d_folders,
        val_image3d_folders=cfg.data.val_image3d_folders,
        val_image2d_folders=cfg.data.val_image2d_folders,
        test_image3d_folders=cfg.data.test_image3d_folders,
        test_image2d_folders=cfg.data.test_image2d_folders,
        img_shape=cfg.data.img_shape,
        vol_shape=cfg.data.vol_shape,
        batch_size=cfg.data.batch_size,
        train_samples=cfg.data.train_samples,
        val_samples=cfg.data.val_samples,
        test_samples=cfg.data.test_samples,
    )

    model = NVLightningModule(model_cfg=cfg.model, train_cfg=cfg.train,)
    callbacks = [hydra.utils.instantiate(c) for c in cfg.callbacks]
    logger = [hydra.utils.instantiate(c) for c in cfg.logger]

    trainer = Trainer(callbacks=callbacks, logger=logger, **cfg.trainer)

    trainer.fit(
        model,
        # datamodule=datamodule,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
        # ckpt_path=cfg.resume_from_checkpoint
    )


if __name__ == "__main__":
    main()
