import os
import logging as log
from typing import Tuple
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import trimesh
import mcubes
import wandb

from .ray import exponential_integration
from ..utils.metrics import psnr
# from trimesh.smoothing import filter_taubin


# Warning: you MUST NOT change the resolution of marching cube
RES = 256

def generate_deltas(ts: torch.Tensor):
    """Calculates the difference between each 'time' in ray samples.

    Rays will go to infinity unless obstructed. Therefore, the delta
    between the ts and infinity is expressed as 1e10

    Args:
        ts: [B x N x num_samples x 1] tensor of times. The values are increasing from [near,far] along
            the num_samples dimension.
    Returns:
        deltas: [B x N x num_samples x 1]  where delta_i = t_i+1 - t_i.
    """
    B, N, _, _ = ts.shape
    deltas = torch.cat([ts[:, :, 1:, :] - ts[:, :, :-1, :], torch.full((B, N, 1, 1), 1e10, device=ts.device)], dim=2)
    return deltas

def inverse_transform_sampling(ray_orig: torch.Tensor, ray_dir: torch.Tensor, weights, ts,
                               num_points: int=128, near=1.0, far=3.0):
    """Performs inverse transform sampling according to the weights.

    Samples from ts according to the weights (i.e. ts with higher weights are 
    more likely to be sampled).
    
    Probably not the best implementation, since the official NeRF implementation 
    does something different. This is probably good enough though? Good thing
    I don't have to be rigorous. 

    Args:
        ray_orig: [B x Nr x 3] coordinates of the ray origin.
        ray_dir: [B x Nr x 3] directions of the ray.
        weights: [B x Nr x Np x 1] tensor of weights calculated as 
                 w = T(1 - exp(- density * delta)). N is the batch size, and C 
                 is the number of coarse samples.
        ts: [B x Nr x Np x 1] is the increment between each sample. N is the batch 
            size, and Np is the number of coarse samples. 
        num_points: number of samples to return per ray.
        near/far: near/far bounds for sampling. 
    Returns:
        fine_samples: [B x Nr x num_points x 3] tensor sampled according to weights.
                      Instead of using the same values as in ts, we pertube it by 
                      adding random noise (sampled from U(0, 1/num_points)).
        fine_ts: [B x Nr x num_points x 1] tensor of the time increment for each sample. 
    """
    device = ray_orig.device
    B, N, C, _ = ts.shape

    cdf = torch.cumsum(weights, axis=2)  # [B x N x C x 1]
    cdf = cdf / cdf[:, -1, None]
    eps = torch.rand((N, 1), device=device) / num_points  # low variance sampling
    samples = torch.arange(0, 1, 1 / num_points, device=device)
    samples = torch.broadcast_to(samples, (B, N, num_points))
    samples = samples + eps
    cdf = torch.squeeze(cdf, -1)  # make dimensions match, [B x N x C]
    lower_idxs = torch.searchsorted(cdf, samples).unsqueeze(-1)  # [B x N x C x 1]
    upper_idxs = lower_idxs + 1

    lower = torch.full((B, N, 1, 1), near, device=device)
    upper = torch.full((B, N, 1, 1), far, device=device)
    ts_bounds = torch.cat([lower, ts, upper], dim=2)

    lower_bins = torch.gather(ts_bounds, 2, lower_idxs)
    upper_bins = torch.gather(ts_bounds, 2, upper_idxs)

    fine_ts = lower_bins + (upper_bins - lower_bins) * torch.rand((B, N, num_points, 1), device=device)
    fine_samples = ray_orig.unsqueeze(2) + fine_ts * ray_dir.unsqueeze(2)

    # Combine coarse and fine samples
    coarse_coords = ray_orig.unsqueeze(2) + ts * ray_dir.unsqueeze(2)
    fine_coords = torch.cat([fine_samples, coarse_coords], dim=2)
    fine_z_vals = torch.cat([fine_ts, ts], dim=2)
    fine_z_vals, idxs = torch.sort(fine_z_vals, dim=2)
    fine_coords = torch.gather(fine_coords, 2, idxs.expand(-1, -1, -1, 3))
    # fine_deltas = generate_deltas(fine_z_vals)

    fine_deltas = fine_z_vals.squeeze(-1).diff(
        dim=-1,
        prepend=(torch.zeros(B, ray_orig.shape[1], 1, device=fine_z_vals.device) + near))

    # t_dists = fine_z_vals[..., 1:] - fine_z_vals[..., :-1]  # Shape: [batch_size, Nr, num_samples]
    # fine_deltas = t_dists * torch.linalg.norm(ray_dir[..., None, :], dim=-1)
    
    fine_deltas = fine_deltas[..., None]
    # fine_deltas = generate_deltas(fine_z_vals)
    
    return fine_coords, fine_z_vals, fine_deltas

class Trainer(nn.Module):

    def __init__(self, config, coarse_model, fine_model, pe_pos, pe_ray_dir, log_dir):
        super().__init__()

        self.cfg = config
        self.pe_pos = pe_pos.cuda()
        self.pe_ray_dir = pe_ray_dir.cuda()
        self.coarse_model = coarse_model.cuda()
        self.fine_model = fine_model.cuda()

        self.log_dir = log_dir
        self.log_dict = {}

        self.init_optimizer()
        self.init_log_dict()

    def init_optimizer(self):
        self.optimizer_coarse = torch.optim.Adam(self.coarse_model.parameters(), lr=self.cfg.lr_start, 
                                    betas=(self.cfg.beta1, self.cfg.beta2),
                                    weight_decay=self.cfg.weight_decay)
        self.optimizer_fine = torch.optim.Adam(self.fine_model.parameters(), lr=self.cfg.lr_start, 
                                    betas=(self.cfg.beta1, self.cfg.beta2),
                                    weight_decay=self.cfg.weight_decay)
        gamma = (self.cfg.lr_end / self.cfg.lr_start) ** (1 / self.cfg.epochs)
        gamma = 0.9997
        self.scheduler_coarse = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_coarse, gamma=gamma)
        self.scheduler_fine = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_fine, gamma=gamma)

    def init_log_dict(self):
        """Custom log dict.
        """
        self.log_dict['total_loss'] = 0.0
        self.log_dict['rgb_loss'] = 0.0
        self.log_dict['total_iter_count'] = 0
        self.log_dict['image_count'] = 0

    def sample_points(self, ray_orig, ray_dir, near=1.0, far=3.0, num_points=64):
        """Sample points along rays. Retruns 3D coordinates of the points.
        TODO: One and extend this function to the hirachical sampling technique 
             used in NeRF or design a more efficient sampling technique for 
             better surface reconstruction.

        Args:
            ray_orig (torch.FloatTensor): Origin of the rays of shape [B, Nr, 3].
            ray_dir (torch.FloatTensor): Direction of the rays of shape [B, Nr, 3].
            near (float): Near plane of the camera.
            far (float): Far plane of the camera.
            num_points (int): Number of points (Np) to sample along the rays.

         Returns:
            points (torch.FloatTensor): 3D coordinates of the points of shape [B, Nr, Np, 3].
            z_vals (torch.FloatTensor): Depth values of the points of shape [B, Nr, Np, 1].
            deltas (torch.FloatTensor): Distance between the points of shape [B, Nr, Np, 1].
        """

        B, Nr = ray_orig.shape[:2]
        t = torch.linspace(0.0, 1.0, num_points, device=ray_orig.device).view(1, 1, -1) + \
            (torch.rand(B, Nr, num_points, device=ray_orig.device)/ num_points)

        z_vals = near * (1.0 - t) + far * t
        points = ray_orig[:, :, None, :] + ray_dir[:, :, None, :] * z_vals[..., None]
        # deltas = generate_deltas(z_vals)
        deltas = z_vals.diff(dim=-1, prepend=(torch.zeros(B, Nr, 1, device=z_vals.device) + near))

        coarse_coords, coarse_z_vals, coarse_deltas = points, z_vals[..., None], deltas[..., None]

        # Step 2 : Predict radiance and volume density at the sampled points
        rgb, sigma = self.predict_radience(coarse_coords, ray_dir, model=self.coarse_model)
            
        # Step 3 : Volume rendering to compute the RGB color at the given rays
        ray_colors, coarse_ray_depth, coarse_ray_alpha, ray_weights = self.volume_render(rgb, sigma, coarse_z_vals, coarse_deltas)
        # Step 4 : Compositing with background color
        if self.cfg.bg_color == 'white':
            bg = torch.ones(B, Nr, 3, device=ray_colors.device)
            coarse_rgb = (1 - coarse_ray_alpha) * bg + coarse_ray_alpha * ray_colors
        else:
            coarse_rgb = coarse_ray_alpha * ray_colors

        fine_coords, fine_z_vals, fine_deltas = inverse_transform_sampling(
            ray_orig, ray_dir, ray_weights, coarse_z_vals, num_points // 2, near, far)
        return fine_coords, fine_z_vals, fine_deltas, coarse_rgb, coarse_ray_depth, coarse_ray_alpha

    def predict_radience(self, coords, ray_dir, model=None):
        """Predict radiance at the given coordinates.
        TODO: You can adjust the network architecture according to your needs. You may also 
        try to use additional raydirections inputs to predict the radiance.

        Args:
        # pos B, Nr, Np, 3
        # dir B, Nr, 3
            coords (torch.FloatTensor): 3D coordinates of the points of shape [..., 3].

        Returns:
            rgb (torch.FloatTensor): Radiance at the given coordinates of shape [..., 3].
            sigma (torch.FloatTensor): volume density at the given coordinates of shape [..., 1].

        """
        if model is None:
            model = self.fine_model
        if len(coords.shape) == 2:
            coords = self.pe_pos(coords)
            ray_dir = self.pe_ray_dir(ray_dir)
        else:
            ray_dir = ray_dir / torch.linalg.norm(ray_dir, dim=-1, keepdim=True)  # unit direction
            ray_dir = ray_dir.unsqueeze(2).repeat(1, 1, coords.shape[-2], 1)
            input_shape = coords.shape
            input_shape_ray_dir = ray_dir.shape

            coords = self.pe_pos(coords.view(-1, 3)).view(*input_shape[:-1], -1)
            ray_dir = self.pe_ray_dir(ray_dir.contiguous().view(-1, 3)).view(*input_shape_ray_dir[:-1], -1)
        rgb, sigma = model(coords, ray_dir)

        return rgb, sigma

    def volume_render(self, rgb, sigma, depth, deltas):
        """Ray marching to compute the radiance at the given rays.
        TODO: You are free to try out different neural rendering methods.
        
        Args:
            rgb (torch.FloatTensor): Radiance at the sampled points of shape [B, Nr, Np, 3].
            sigma (torch.FloatTensor): Volume density at the sampled points of shape [B, Nr, Np, 1].
            deltas (torch.FloatTensor): Distance between the points of shape [B, Nr, Np, 1].
        
        Returns:
            ray_colors (torch.FloatTensor): Radiance at the given rays of shape [B, Nr, 3].
            weights (torch.FloatTensor): Weights of the given rays of shape [B, Nr, 1].

        """
        # Sample points along the rays

        tau = sigma * deltas
        ray_colors, ray_depth, ray_alpha, weights = exponential_integration(rgb, tau, depth, exclusive=True)
        return ray_colors, ray_depth, ray_alpha, weights

    def forward(self):
        """Forward pass of the network. 
        TODO: Adjust the neural rendering pipeline according to your needs.

        Returns:
            rgb (torch.FloatTensor): Ray codors of shape [B, Nr, 3].

        """
        B, Nr = self.ray_orig.shape[:2]
        ray_dir = self.ray_dir
        # Step 1 : Sample points along the rays
        fine_coords, fine_z_vals, fine_deltas, coarse_rgb, coarse_ray_depth, coarse_ray_alpha = self.sample_points(
                                self.ray_orig, ray_dir, near=self.cfg.near, far=self.cfg.far,
                                num_points=self.cfg.num_pts_per_ray)

        self.coarse_rgb = coarse_rgb
        self.coarse_ray_depth = coarse_ray_depth
        self.coarse_ray_alpha = coarse_ray_alpha

        rgb, sigma = self.predict_radience(fine_coords, ray_dir, model=self.fine_model)
        ray_colors, ray_depth, ray_alpha, _ = self.volume_render(rgb, sigma, fine_z_vals, fine_deltas)

        self.fine_ray_depth = ray_depth
        self.fine_ray_alpha = ray_alpha

        # Step 4 : Compositing with background color
        if self.cfg.bg_color == 'white':
            bg = torch.ones(B, Nr, 3, device=ray_colors.device)
            self.fine_rgb = (1 - ray_alpha) * bg + ray_alpha * ray_colors
        else:
            self.fine_rgb = ray_alpha * ray_colors

    def backward(self):
        """Backward pass of the network.
        TODO: You can also desgin your own loss function.
        """

        loss = 0.0
        fine_loss = torch.nn.MSELoss()(self.fine_rgb, self.img_gts)
        coarse_loss = torch.nn.MSELoss()(self.coarse_rgb, self.img_gts)

        rgb_loss = fine_loss

        loss = rgb_loss

        self.log_dict['rgb_loss'] += rgb_loss.item()
        self.log_dict['total_loss'] += loss.item()

        coarse_loss.backward(retain_graph=True)
        fine_loss.backward()

    def step(self, data):
        """A signle training step.
        """

        # Get rays, and put them on the device
        self.ray_orig = data['rays'][..., :3].cuda()
        self.ray_dir = data['rays'][..., 3:].cuda()
        self.img_gts = data['imgs'].cuda()

        self.optimizer_coarse.zero_grad()
        self.optimizer_fine.zero_grad()

        self.forward()
        self.backward()

        self.optimizer_coarse.step()
        self.optimizer_fine.step()
        self.log_dict['total_iter_count'] += 1
        self.log_dict['image_count'] += self.ray_orig.shape[0]

    def render(self, ray_orig, ray_dir):
        """Render a full image for evaluation.
        """
        B, Nr = ray_orig.shape[:2]
        coords, depth, deltas, _, _, _ = self.sample_points(ray_orig, ray_dir, near=self.cfg.near, far=self.cfg.far,
                                num_points=self.cfg.num_pts_per_ray_render)
        rgb, sigma = self.predict_radience(coords, ray_dir)
        ray_colors, ray_depth, ray_alpha, _ = self.volume_render(rgb, sigma, depth, deltas)

        if self.cfg.bg_color == 'white':
            bg = torch.ones(B, Nr, 3, device=ray_colors.device)
            render_img = (1 - ray_alpha) * bg + ray_alpha * ray_colors
        else:
            render_img = ray_alpha * ray_colors

        return render_img, ray_depth, ray_alpha

    def reconstruct_3D(self, save_dir, epoch=0, sigma_threshold = 50., chunk_size=8192, smoothing_sigma=1.0):
        """Reconstruct the 3D shape from the volume density.
        """

        # Mesh evaluation
        window_x = torch.linspace(-1., 1., steps=RES, device='cuda')
        window_y = torch.linspace(-1., 1., steps=RES, device='cuda')
        window_z = torch.linspace(-1., 1., steps=RES, device='cuda')
        
        coord = torch.stack(torch.meshgrid(window_x, window_y, window_z)).permute(1, 2, 3, 0).reshape(-1, 3).contiguous()
        # torch.tensor([0.0, 0.0, 1.0], device=coords.device).view(1, 3).expand(coords.shape[0], 3)
        ray_dir = torch.zeros_like(coord)
        _points = torch.split(coord, int(chunk_size), dim=0)
        _ray_dirs = torch.split(ray_dir, int(chunk_size), dim=0)

        voxels = []
        for _p, _ray_dir in zip(_points, _ray_dirs):
            _, sigma = self.predict_radience(_p, _ray_dir)
            voxels.append(sigma)
        voxels = torch.cat(voxels, dim=0)
        np_sigma = torch.clip(voxels, 0.0).reshape(RES, RES, RES).cpu().numpy()

        vertices, faces = mcubes.marching_cubes(np_sigma, 50)
        #vertices = ((vertices - 0.5) / (res/2)) - 1.0
        vertices = (vertices / (RES-1)) * 2.0 - 1.0

        h = trimesh.Trimesh(vertices=vertices, faces=faces)

        # h = h.simplify_quadratic_decimation(int(len(h.faces) * 0.9))

        h.export(os.path.join(save_dir, '%04d.obj' % (epoch)))

    def log(self, step, epoch):
        """Log the training information.
        """
        log_text = 'STEP {} - EPOCH {}/{}'.format(step, epoch, self.cfg.epochs)
        self.log_dict['total_loss'] /= self.log_dict['total_iter_count']
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'])
        self.log_dict['rgb_loss'] /= self.log_dict['total_iter_count']
        log_text += ' | rgb loss: {:>.3E}'.format(self.log_dict['rgb_loss'])

        log.info(log_text)

        for key, value in self.log_dict.items():
            if 'loss' in key:
                wandb.log({key: value}, step=step)
        self.init_log_dict()

    def validate(self, loader, img_shape, step=0, epoch=0, sigma_threshold = 50., chunk_size=8192, save_img=False):
        """validation function for generating final results.
        """
        torch.cuda.empty_cache() # To avoid CUDA out of memory
        self.eval()

        log.info("Beginning validation...")
        log.info(f"Loaded validation dataset with {len(loader)} images at resolution {img_shape[0]}x{img_shape[1]}")


        self.valid_mesh_dir = os.path.join(self.log_dir, "mesh")
        log.info(f"Saving reconstruction result to {self.valid_mesh_dir}")
        if not os.path.exists(self.valid_mesh_dir):
            os.makedirs(self.valid_mesh_dir)

        if save_img:
            self.valid_img_dir = os.path.join(self.log_dir, "img")
            log.info(f"Saving rendering result to {self.valid_img_dir}")
            if not os.path.exists(self.valid_img_dir):
                os.makedirs(self.valid_img_dir)

        psnr_total = 0.0

        wandb_img = []
        wandb_img_gt = []

        with torch.no_grad():
            # Evaluate 3D reconstruction
            self.reconstruct_3D(self.valid_mesh_dir, epoch=epoch,
                            sigma_threshold=sigma_threshold, chunk_size=chunk_size)

            # Evaluate 2D novel view rendering
            for i, data in enumerate(tqdm(loader)):
                rays = data['rays'].cuda()          # [1, Nr, 6]
                img_gt = data['imgs'].cuda()        # [1, Nr, 3]
                mask = data['masks'].repeat(1, 1, 3).cuda()

                _rays = torch.split(rays, int(chunk_size), dim=1)
                pixels = []
                for _r in _rays:
                    ray_orig = _r[..., :3]          # [1, chunk, 3]
                    ray_dir = _r[..., 3:]           # [1, chunk, 3]
                    ray_rgb, ray_depth, ray_alpha = self.render(ray_orig, ray_dir)
                    pixels.append(ray_rgb)

                pixels = torch.cat(pixels, dim=1)

                psnr_total += psnr(pixels, img_gt)

                img = (pixels).reshape(*img_shape, 3).cpu().numpy() * 255
                gt = (img_gt).reshape(*img_shape, 3).cpu().numpy() * 255
                wandb_img.append(wandb.Image(img))
                wandb_img_gt.append(wandb.Image(gt))

                if save_img:
                    Image.fromarray(gt.astype(np.uint8)).save(
                        os.path.join(self.valid_img_dir, "gt-{:04d}-{:03d}.png".format(epoch, i)) )
                    Image.fromarray(img.astype(np.uint8)).save(
                        os.path.join(self.valid_img_dir, "img-{:04d}-{:03d}.png".format(epoch, i)) )

        wandb.log({"Rendered Images": wandb_img}, step=step)
        wandb.log({"Ground-truth Images": wandb_img_gt}, step=step)
                
        psnr_total /= len(loader)

        log_text = 'EPOCH {}/{}'.format(epoch, self.cfg.epochs)
        log_text += ' {} | {:.2f}'.format(f"PSNR", psnr_total)

        wandb.log({'PSNR': psnr_total, 'Epoch': epoch}, step=step)
        log.info(log_text)
        self.train()

    def save_model(self, epoch):
        """Save the model checkpoint.
        """

        fine_fname = os.path.join(self.log_dir, f'fine_model-{epoch}.pth')
        coarse_fname = os.path.join(self.log_dir, f'coarse_model-{epoch}.pth')
        log.info(f'Saving model checkpoint to: {fine_fname}')
        torch.save(self.coarse_model, coarse_fname)
        torch.save(self.fine_model, fine_fname)


# first shell is generate_delta
# second is from mp untouched.