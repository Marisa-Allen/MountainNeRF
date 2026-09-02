from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Optional
import torch
import torch.nn.functional as F

from nerfstudio.models.depth_nerfacto import DepthNerfactoModel, DepthNerfactoModelConfig
from .dem_field import DEMHeightField

@dataclass
class HeightDepthNerfactoModelConfig(DepthNerfactoModelConfig):
    # Provides all hyperparameter values
    
    _target: Type = field(default_factory=lambda: HeightDepthNerfactoModel)
    # sacling factor for the height loss
    height_loss_mult: float = 0.0001
    # start of ramp
    height_start_ramp: int = 6000
    # lenght of ramp
    height_ramp_steps: int = 4000
    # Path to the DEM
    dem_path: Path = Path("output_AW3D30.tif")
    # Center of the DEM
    dem_lon0: float = -70.63597222220702
    dem_lat0: float = -35.04541666668361
    # floter corrector scaling factor
    floater_loss_mult: float = 0.00002
    # Margin above DEM that accepted
    floater_margin: float = 50
    # Start of ramp
    floater_start_ramp: int = 2000
    # lenght of ramp
    floater_ramp_steps: int = 5000

class DataparserInverseTransform:
# Inverse transforms what the NeRF does when traning.
    
    def __init__(self, transform, scale, device="cuda"):
        transform = torch.as_tensor(transform, dtype=torch.float32, device=device)
        # rotation
        self.R = transform[:, :3]
        # translation
        self.t = transform[:, 3]
        # scale
        self.scale = float(scale)
 
    def to_dem_frame(self, p_normalized: torch.Tensor) -> torch.Tensor:
        # point
        p = p_normalized / self.scale - self.t
        # inverse transforming
        result = p @ self.R
        result[..., 2] *= -1
        return result

# code for model MountianNeRF
class MountainNeRF(DepthNerfactoModel):
    config: HeightDepthNerfactoModelConfig

    def populate_modules(self):
        # setting up
        super().populate_modules()
        self.dem_field = DEMHeightField(
            str(self.config.dem_path),
            lon0=self.config.dem_lon0,
            lat0=self.config.dem_lat0,
        )
        self._last_ray_bundle = None 
        # start at step 0
        self.step = 0

        # transformation - unique to each nerf enviroment
        self.dataparser_inv = DataparserInverseTransform(
            transform=[
                [0.00019717200484592468, 0.9999753832817078, 0.007013166788965464, -7221.35302734375],
                [0.9984204173088074, 0.00019717200484592468, -0.056183986365795135, 1445.6441650390625],
                [-0.056183986365795135, 0.007013166788965464, -0.9983958005905151, -2035.5819091796875],
            ],
            scale=0.00014562754021574207,
        )

    def get_outputs(self, ray_bundle):
        outputs = super().get_outputs(ray_bundle)
        self._last_ray_bundle = ray_bundle

        if "weights_list" in outputs and "ray_samples_list" in outputs:
            self._last_weights = outputs["weights_list"][-1]
            self._last_ray_samples = outputs["ray_samples_list"][-1]
        else:
            self._last_weights = None
            self._last_ray_samples = None

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
    # getting the new losses for MountainNeRF
    
        self.step += 1
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # Height loss
        ray_bundle = self._last_ray_bundle
        if ray_bundle is None or "expected_depth" not in outputs:
            return loss_dict
        
        # getting expected depth
        exp_depth = outputs["expected_depth"]
        accum = outputs["accumulation"]

        # raybunforgien - camera/ray oriigen, 
        # xyz_hat - predicted 3d point
        # need to transform xyzhat
        xyz_nerf = ray_bundle.origins + exp_depth * ray_bundle.directions

        # inversing from NeRF frame
        xyz = self.dataparser_inv.to_dem_frame(xyz_nerf)
        # finding z component
        z_hat = xyz[..., 2]

        # ground truth z from the DEM
        z_dem, valid = self.dem_field(xyz[..., :2])
        
        depth_flat = exp_depth.squeeze(-1)
        # filtering out bad values
        valid_depth = torch.isfinite(depth_flat) & (depth_flat > 0) & (depth_flat < self.config.far_plane)
        # creating mask
        mask = valid & (accum.squeeze(-1) > 0.5) & valid_depth

        # if wihtin restraints of the mask and after the ramp
        if mask.any() and self.step > self.config.height_start_ramp:
            height_loss = F.smooth_l1_loss(z_hat[mask], z_dem[mask])
            # scaling with the ramp 
            ramp = min(1.0, (self.step - self.config.height_start_ramp) / self.config.height_ramp_steps)
            # appending to the dictionary
            loss_dict["height_loss"] = ramp * self.config.height_loss_mult * height_loss
            
        ##############################################################################################
        # floater supressionloss

        # getting weights and positions
        weights = self._last_weights.squeeze(-1)
        poss_nerf = self._last_ray_samples.frustums.get_positions()
        # inverting
        poss = self.dataparser_inv.to_dem_frame(poss_nerf)
        # getting x and y of samples
        xy = poss[..., :2].reshape(-1, 2)

        # z samples
        z_samps, val_samps = self.dem_field(xy)
        z_samps = z_samps.reshape(poss.shape[:2])
        # valid z samples
        val_samps = val_samps.reshape(poss.shape[:2])
        # mask for vlaues that are finite
        finite = torch.isfinite(poss[..., 2]) & torch.isfinite(z_samps)

        # points above DEM
        above_dem = (poss[..., 2] - z_samps) - self.config.floater_margin

        # Applying the penatly
        penalty = torch.relu(above_dem) * val_samps.float() * finite.float()
        #calculating loss
        floaters_loss = (weights * penalty).mean()

        # applying mask and the ramp
        if torch.isfinite(floaters_loss) and self.step > self.config.floater_start_ramp:
            # scaling to ramp and appending to dict
            ramp = min(1.0, (self.step - self.config.floater_start_ramp) / self.config.floater_ramp_steps)
            loss_dict["floaters_loss"] = ramp * self.config.floater_loss_mult * floaters_loss

        return loss_dict
