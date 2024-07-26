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

# from monai.networks.nets import Unet
# from monai.networks.layers.factories import Norm
# from generative.networks.nets import DiffusionModelUNet
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
from monai.utils import optional_import
tqdm, has_tqdm = optional_import("tqdm", name="tqdm")

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


class DiffusionProjectorInferer(DiffusionInferer):
    def __init__(self, 
                scheduler: nn.Module, 
                renderer: nn.Module,
                model_cfg: DictConfig) -> None:
        super().__init__(scheduler)
        self.scheduler = scheduler
        self.renderer = renderer
        self.model_cfg = model_cfg
    
    def __call__(
        self,
        inputs: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
        cam: FoVPerspectiveCameras | None = None,
        mode: str = "crossattn",
        return_volume = False, 
    ) -> torch.Tensor:
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        noisy_image = self.scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps)
        if mode == "concat":
            noisy_image = torch.cat([noisy_image, condition], dim=1)
            condition = None
        out = diffusion_model(x=noisy_image, timesteps=timesteps, context=condition)
        z = torch.linspace(-1.0, 1.0, steps=self.model_cfg.vol_shape, device=noisy_image.device)
        y = torch.linspace(-1.0, 1.0, steps=self.model_cfg.vol_shape, device=noisy_image.device)
        x = torch.linspace(-1.0, 1.0, steps=self.model_cfg.vol_shape, device=noisy_image.device)
        grd = torch.stack(torch.meshgrid(x, y, z), dim=-1).view(-1, 3).unsqueeze(0).repeat(noisy_image.shape[0], 1, 1)  # 1 DHW 3 to B DHW 3
        
        # Process (resample) the volumes from ray views to ndc
        pts = cam.transform_points_ndc(grd)  # world to ndc, 1 DHW 3
        vol = F.grid_sample(
            out.float().unsqueeze(1), 
            pts.view(-1, self.model_cfg.vol_shape, self.model_cfg.vol_shape, self.model_cfg.vol_shape, 3).float(), 
            mode="bilinear", 
            padding_mode="zeros", 
            align_corners=True,
        )
        vol = torch.permute(vol, [0, 1, 4, 2, 3])
        if return_volume:
            return vol
        else:
            prj = self.renderer(vol, cam)
            return prj

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        cam: FoVPerspectiveCameras | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        if not scheduler:
            scheduler = self.scheduler
        image = input_noise
        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)
        intermediates = []
        for t in progress_bar:
            # 1. predict noise model_output
            if mode == "concat":
                model_input = torch.cat([image, conditioning], dim=1)
                model_output = diffusion_model(
                    model_input, 
                    timesteps=torch.Tensor((t,)).to(input_noise.device), 
                    context=None, 
                )
            else:
                model_output = diffusion_model(
                    image, 
                    timesteps=torch.Tensor((t,)).to(input_noise.device), 
                    context=conditioning, 
                )
            z = torch.linspace(-1.0, 1.0, steps=self.model_cfg.vol_shape, device=image.device)
            y = torch.linspace(-1.0, 1.0, steps=self.model_cfg.vol_shape, device=image.device)
            x = torch.linspace(-1.0, 1.0, steps=self.model_cfg.vol_shape, device=image.device)
            grd = torch.stack(torch.meshgrid(x, y, z), dim=-1).view(-1, 3).unsqueeze(0).repeat(image.shape[0], 1, 1)  # 1 DHW 3 to B DHW 3
            
            # Process (resample) the volumes from ray views to ndc
            pts = cam.transform_points_ndc(grd)  # world to ndc, 1 DHW 3
            vol = F.grid_sample(
                model_output.float().unsqueeze(1), 
                pts.view(-1, self.model_cfg.vol_shape, self.model_cfg.vol_shape, self.model_cfg.vol_shape, 3).float(), 
                mode="bilinear", 
                padding_mode="zeros", 
                align_corners=True,
            )
            vol = torch.permute(vol, [0, 1, 4, 2, 3])

            model_output = self.renderer(vol, cam) #Projection

            # 2. compute previous image: x_t -> x_t-1
            image, _ = scheduler.step(model_output, t, image)
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)
        if save_intermediates:
            return image, intermediates
        else:
            return image
        
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
        )

        # @ Diffusion 
        self.unet3d_model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=self.model_cfg.fov_depth,
            num_channels=[256, 256, 512],
            attention_levels=[False, False, True],
            num_head_channels=[0, 0, 512],
            num_res_blocks=2,
            with_conditioning=True, 
            cross_attention_dim=12, # Condition with straight/hidden view  # flatR | flatT
        )
        init_weights(self.unet3d_model, init_type="normal")
        
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
        
        self.inferer = DiffusionProjectorInferer(
            scheduler=self.ddimsch, 
            renderer=self.fwd_renderer, 
            model_cfg=self.model_cfg
        )

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
        image2d = self.fwd_renderer(image3d, cameras, norm_type="standardized", stratified_sampling=is_training)
        return image2d
    
    def flatten_cameras(self, cameras, zero_translation=False):
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
        timesteps = torch.zeros((B,), device=_device).long()if timesteps is None else timesteps

        mat = self.flatten_cameras(cameras, zero_translation=False)

        out = self.inferer(
            inputs=image2d, 
            noise=noise, 
            diffusion_model=self.unet3d_model, 
            condition=mat.view(-1, 1, 12), 
            timesteps=timesteps,
            cam=cameras,
            return_volume=True,
        ) 
        return out
    
    def forward_timing(self, image2d, cameras, noise=None, timesteps=None):
        _device = image2d.device
        B = image2d.shape[0]
        timesteps = torch.zeros((B,), device=_device).long()if timesteps is None else timesteps

        mat = self.flatten_cameras(cameras, zero_translation=False)

        out = self.inferer(
            inputs=image2d, 
            noise=noise, 
            diffusion_model=self.unet3d_model, 
            condition=mat.view(-1, 1, 12), 
            timesteps=timesteps,
            cam=cameras,
            return_volume=False,
        ) 
        return out
    
    def _common_step(self, batch, batch_idx, stage: Optional[str] = "evaluation"):
        image2d = batch["image2d"]
        image3d = batch["image3d"]
        _device = batch["image3d"].device
        B = image2d.shape[0]

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
        figure_ct_source_hidden = self.forward_screen(image3d=image3d, cameras=view_hidden)
        figure_ct_source_random = self.forward_screen(image3d=image3d, cameras=view_random)
        figure_ct_source_second = self.forward_screen(image3d=image3d, cameras=view_second)

        # timesteps = torch.randint(0, self.ddpmsch.num_train_timesteps, (B,), device=_device).long()  
        timesteps = None
        # if stage=='train':
        #     # Generate a random number between 0 and 1
        #     prob = torch.rand(1).item()
        #     if prob < 0.5:
        #         timesteps = torch.randint(0, self.ddpmsch.num_train_timesteps, (B,), device=_device).long()  
            
        # timesteps = None   
        # figure_xr_latent_hidden = torch.randn_like(figure_xr_source_hidden)
        # figure_ct_latent_hidden = torch.randn_like(figure_ct_source_hidden)
        # figure_ct_latent_random = torch.randn_like(figure_ct_source_random)
        # figure_ct_latent_second = torch.randn_like(figure_ct_source_second)
        
        # figure_xr_latent_hidden = self.forward_screen(image3d=torch.randn_like(image3d), cameras=view_hidden)
        # figure_ct_latent_hidden = self.forward_screen(image3d=torch.randn_like(image3d), cameras=view_hidden)
        # figure_ct_latent_random = self.forward_screen(image3d=torch.randn_like(image3d), cameras=view_random)
        # figure_ct_latent_second = self.forward_screen(image3d=torch.randn_like(image3d), cameras=view_second)  

        # # Run the forward pass
        # figure_dx_latent_concat = torch.cat([figure_xr_latent_hidden, figure_ct_latent_hidden, figure_ct_latent_random, figure_ct_latent_second])
        # figure_dx_source_concat = torch.cat([figure_xr_source_hidden, figure_ct_source_hidden, figure_ct_source_random, figure_ct_source_second])
        # camera_dx_render_concat = join_cameras_as_batch([view_hidden, view_hidden, view_random, view_second])

        # # For 3D with latent
        # volume_dx_reproj_concat = self.forward_volume(
        #     image2d=figure_dx_source_concat, 
        #     cameras=camera_dx_render_concat, 
        #     noise=figure_dx_latent_concat, 
        #     timesteps=timesteps,
        # )

        # Run the forward pass
        figure_dx_source_concat = torch.cat([figure_xr_source_hidden, figure_ct_source_hidden, figure_ct_source_random, figure_ct_source_second])
        camera_dx_render_concat = join_cameras_as_batch([view_hidden, view_hidden, view_random, view_second])

        # For 3D
        volume_dx_reproj_concat = self.forward_volume(
            image2d=figure_dx_source_concat, 
            cameras=camera_dx_render_concat, 
            noise=torch.zeros_like(figure_dx_source_concat), 
            timesteps=None,
        )
        volume_xr_reproj_hidden, \
        volume_ct_reproj_hidden, \
        volume_ct_reproj_random, \
        volume_ct_reproj_second = torch.split(volume_dx_reproj_concat, B, dim=0)
        
        figure_xr_reproj_hidden_hidden = self.forward_screen(image3d=volume_xr_reproj_hidden, cameras=view_hidden)
        figure_xr_reproj_hidden_random = self.forward_screen(image3d=volume_xr_reproj_hidden, cameras=view_random)
        figure_xr_reproj_hidden_second = self.forward_screen(image3d=volume_xr_reproj_hidden, cameras=view_second)
        
        figure_xr_reproj_concat = torch.cat([figure_xr_reproj_hidden_hidden, figure_xr_reproj_hidden_random, figure_xr_reproj_hidden_second])
        camera_xr_render_concat = join_cameras_as_batch([view_hidden, view_random, view_second])

        volume_xr_reproj_concat = self.forward_volume(
            image2d=figure_xr_reproj_concat, 
            cameras=camera_xr_render_concat, 
            noise=torch.zeros_like(figure_xr_reproj_concat), 
            timesteps=None,
        )

        volume_xr_reproj_hidden_reproj, \
        volume_xr_reproj_random_reproj, \
        volume_xr_reproj_second_reproj = torch.split(volume_xr_reproj_concat, B, dim=0)

        figure_ct_reproj_hidden_hidden = self.forward_screen(image3d=volume_ct_reproj_hidden, cameras=view_hidden)
        figure_ct_reproj_hidden_random = self.forward_screen(image3d=volume_ct_reproj_hidden, cameras=view_random)
        figure_ct_reproj_hidden_second = self.forward_screen(image3d=volume_ct_reproj_hidden, cameras=view_second)
        
        figure_ct_reproj_random_hidden = self.forward_screen(image3d=volume_ct_reproj_random, cameras=view_hidden)
        figure_ct_reproj_random_random = self.forward_screen(image3d=volume_ct_reproj_random, cameras=view_random)
        figure_ct_reproj_random_second = self.forward_screen(image3d=volume_ct_reproj_random, cameras=view_second)
        
        figure_ct_reproj_second_hidden = self.forward_screen(image3d=volume_ct_reproj_second, cameras=view_hidden)
        figure_ct_reproj_second_random = self.forward_screen(image3d=volume_ct_reproj_second, cameras=view_random)
        figure_ct_reproj_second_second = self.forward_screen(image3d=volume_ct_reproj_second, cameras=view_second)

        im3d_loss_inv = F.l1_loss(volume_ct_reproj_hidden, image3d) \
                      + F.l1_loss(volume_ct_reproj_random, image3d) \
                      + F.l1_loss(volume_ct_reproj_second, image3d) \
                      + F.l1_loss(volume_xr_reproj_hidden_reproj, volume_xr_reproj_hidden) \
                      + F.l1_loss(volume_xr_reproj_random_reproj, volume_xr_reproj_hidden) \
                      + F.l1_loss(volume_xr_reproj_second_reproj, volume_xr_reproj_hidden)
        
        im2d_loss_inv = F.l1_loss(figure_ct_reproj_hidden_hidden, figure_ct_source_hidden) \
                      + F.l1_loss(figure_ct_reproj_hidden_random, figure_ct_source_random) \
                      + F.l1_loss(figure_ct_reproj_hidden_second, figure_ct_source_second) \
                      + F.l1_loss(figure_ct_reproj_random_hidden, figure_ct_source_hidden) \
                      + F.l1_loss(figure_ct_reproj_random_random, figure_ct_source_random) \
                      + F.l1_loss(figure_ct_reproj_random_second, figure_ct_source_second) \
                      + F.l1_loss(figure_ct_reproj_second_hidden, figure_ct_source_hidden) \
                      + F.l1_loss(figure_ct_reproj_second_random, figure_ct_source_random) \
                      + F.l1_loss(figure_ct_reproj_second_second, figure_ct_source_second) 
        
        pc3d_loss_all = self.p3dloss(volume_ct_reproj_hidden, image3d) \
                      + self.p3dloss(volume_ct_reproj_random, image3d) \
                      + self.p3dloss(volume_ct_reproj_second, image3d) \
                      + self.p3dloss(volume_xr_reproj_hidden, image3d) 
        
        pc2d_loss_all = self.p2dloss(figure_ct_reproj_hidden_hidden, figure_ct_source_hidden) \
                      + self.p2dloss(figure_ct_reproj_hidden_random, figure_ct_source_random) \
                      + self.p2dloss(figure_ct_reproj_hidden_second, figure_ct_source_second) \
                      + self.p2dloss(figure_ct_reproj_random_hidden, figure_ct_source_hidden) \
                      + self.p2dloss(figure_ct_reproj_random_random, figure_ct_source_random) \
                      + self.p2dloss(figure_ct_reproj_random_second, figure_ct_source_second) \
                      + self.p2dloss(figure_ct_reproj_second_hidden, figure_ct_source_hidden) \
                      + self.p2dloss(figure_ct_reproj_second_random, figure_ct_source_random) \
                      + self.p2dloss(figure_ct_reproj_second_second, figure_ct_source_second) \
                      + self.p2dloss(figure_xr_reproj_hidden_random, figure_ct_source_random) \
                      + self.p2dloss(figure_xr_reproj_hidden_second, figure_ct_source_second) \
                      + self.p2dloss(figure_xr_reproj_hidden_hidden, image2d) 
                      
        loss = self.train_cfg.alpha * im2d_loss_inv + self.train_cfg.gamma * im3d_loss_inv \
             + self.train_cfg.lamda * pc2d_loss_all + self.train_cfg.lamda * pc3d_loss_all  
    
        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=B)
        
        # Visualization step
        if batch_idx == 0:
            # Sampling step for X-ray
            with torch.no_grad():
                zeros = torch.zeros_like(image2d)
                
                viz2d = torch.cat([
                    torch.cat([
                        figure_xr_source_hidden, 
                        figure_ct_source_hidden, 
                        figure_ct_source_random, 
                        figure_ct_source_second,
                        figure_xr_reproj_hidden_random,
                        image3d[..., self.model_cfg.vol_shape // 2, :], 
                        image3d[..., self.model_cfg.vol_shape // 2, :], 
                        image3d[..., self.model_cfg.vol_shape // 2, :], 
                    ], dim=-2).transpose(2, 3),
                    torch.cat([
                        figure_xr_reproj_hidden_hidden, 
                        figure_ct_reproj_hidden_hidden, 
                        figure_ct_reproj_hidden_random, 
                        figure_ct_reproj_hidden_second,
                        volume_xr_reproj_hidden[..., self.model_cfg.vol_shape // 2, :], 
                        volume_ct_reproj_hidden[..., self.model_cfg.vol_shape // 2, :], 
                        volume_ct_reproj_random[..., self.model_cfg.vol_shape // 2, :], 
                        volume_ct_reproj_second[..., self.model_cfg.vol_shape // 2, :], 
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
            optimizer, #
            milestones=[100, 200, 300, 400], 
            gamma=0.5
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
