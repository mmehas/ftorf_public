import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
import tensorflow as tf
import numpy as np
import imageio
import json
import random
import time
import cv2
import pickle

from scipy import ndimage, misc
from matplotlib import cm

from utils.nerf_utils import *
from utils.utils import *
from utils.camera_utils import *
from utils.tof_utils import get_amplitude, get_phase, get_phasor
from utils.flow_utils import compute_flow_metrics

from datasets.tof_dataset import ToFDataset
from datasets.ios_dataset import IOSDataset
from datasets.real_dataset import RealDataset
from datasets.mitsuba_dataset import MitsubaDataset
from datasets.quads_dataset import QuadsDataset

from config import config_parser
from render import *
from losses import *

from PIL import Image
from PIL import ImageDraw, ImageFont

from gradient_accumulator import GradientAccumulateOptimizer

import matplotlib.pyplot as plt


class NeRFTrainer(object):
    def __init__(self, args):
        ## Args
        self.args = args

        ## Logging
        self.setup_loggers()

        ## Dataset
        self.setup_dataset()

        ## Projection, ray generation functions
        self.setup_ray_generation()

        ## Models
        self.setup_models()

        ## Losses
        self.setup_losses()

        ## Training stages
        self.model_reset_done = not self.args.reset_static_model
        self.calibration_pretraining_done = not self.args.calibration_pretraining

    def setup_ray_generation(self):
        ## Projection, ray generation functions
        self.image_sizes = {}
        self.generate_rays = {}

        if self.args.dataset_type == 'quads':
            self.image_sizes['tof'] = (
                self.dataset['tofQuad0_images'].shape[2],
                self.dataset['tofQuad0_images'].shape[1]
            )
        else :    
            self.image_sizes['tof'] = (
                self.dataset['tof_images'].shape[2],
                self.dataset['tof_images'].shape[1]
            )

        self.image_sizes['color'] = (
            self.dataset['color_images'].shape[2],
            self.dataset['color_images'].shape[1]
        )

        self.generate_rays['tof_coords'] = create_ray_generation_coords_fn(
            self.dataset['tof_intrinsics'][0]
            )
        self.generate_rays['color_coords'] = create_ray_generation_coords_fn(
            self.dataset['color_intrinsics'][0]
            )

        self.generate_rays['tof'] = create_ray_generation_fn(
            self.image_sizes['tof'][1],
            self.image_sizes['tof'][0],
            self.dataset['tof_intrinsics'][0]
        )
        self.generate_rays['color'] = create_ray_generation_fn(
            self.image_sizes['color'][1],
            self.image_sizes['color'][0],
            self.dataset['color_intrinsics'][0]
        )

    def set_trainable_pose(self, key):
        self.poses[key] = tf.Variable(
            self.poses[key],
            dtype=tf.float32
        )
        self.grad_calib_vars.append(self.poses[key])
    
    def set_saveable_pose(self, key):
        self.calib_vars[key] = self.poses[key]
        
    def setup_calibration(self):
        self.calib_vars = {}
        self.grad_calib_vars = []

        ## Poses
        self.poses = {}

        if self.args.optimize_poses and self.args.use_relative_poses:
            trainable_pose_names = ['tof_poses']

            if not self.args.collocated_pose:
                trainable_pose_names += ['relative_pose']
        else:
            trainable_pose_names = ['tof_poses', 'color_poses']

        if 'relative_pose' not in self.dataset:
            self.dataset['relative_pose'] = np.eye(4)[None].astype(np.float32)

        for k in self.dataset:
            if 'pose' not in k:
                continue

            self.poses[k] = se3_vee(self.dataset[k])

            if self.args.optimize_poses and (k in trainable_pose_names):
                if self.args.identity_pose_initialization and not k == 'relative_pose':
                    self.poses[k] = np.zeros_like(
                        self.poses[k]
                    )

                if self.args.noisy_pose_initialization and not k == 'relative_pose':
                    self.poses[k] = add_pose_noise(
                        self.poses[k]
                    )

                self.set_trainable_pose(k)

            self.set_saveable_pose(k)

        ## Phase offset
        if 'phase_offset' in self.dataset:
            phase_offset = tf.convert_to_tensor(
                np.array(self.dataset['phase_offset']).astype(np.float32)
            )
        else:
            phase_offset = tf.convert_to_tensor(
                np.array(self.args.phase_offset).astype(np.float32)
            )

        if self.args.optimize_phase_offset:
            phase_offset = tf.Variable(phase_offset, dtype=tf.float32)

            self.calib_vars['phase_offset'] = phase_offset
            self.grad_vars.append(phase_offset)

        # DC Offset
        self.dc_offset = tf.convert_to_tensor(
                np.array(self.args.dc_offset).astype(np.float32)
            )

        self.dc_offset = tf.Variable(self.dc_offset, dtype=tf.float32)
        
        if self.args.optimize_dc_offset:
            self.grad_vars.append(self.dc_offset)

        self.calib_vars['phase_offset'] = phase_offset

        ## Depth range
        if 'depth_range' in self.dataset:
            self.args.depth_range = self.dataset['depth_range']
            self.render_kwargs_train['depth_range'] = self.dataset['depth_range']
            self.render_kwargs_test['depth_range'] = self.dataset['depth_range']
        
        self.emitter_intensity = tf.convert_to_tensor(
            np.array(self.args.emitter_intensity).astype(np.float32)
        )
        self.emitter_intensity = tf.Variable(self.emitter_intensity, dtype=tf.float32)
        if self.args.optimize_emitter_intensity:
            self.grad_vars.append(self.emitter_intensity)
        self.calib_vars['emitter_intensity'] = self.emitter_intensity

    @property
    def relative_pose(self):
        return se3_hat(self.poses['relative_pose'])

    @property
    def tof_poses(self):
        return se3_hat(self.poses['tof_poses'])

    @property
    def tof_light_poses(self):
        return se3_hat(self.poses['tof_poses'])

    @property
    def color_poses(self):
        if self.args.optimize_poses and self.args.use_relative_poses:
            return self.tof_poses @ self.relative_pose
        else:
            return se3_hat(self.poses['color_poses'])

    @property
    def color_light_poses(self):
        return se3_hat(self.poses['color_light_poses'])

    def save_calibration(self, i):
        # Poses
        for k in self.calib_vars:
            calib_path = os.path.join(
                self.basedir, self.expname, '{}_{:06d}.npy'.format(k, i)
            )
            calib_path_full = os.path.join(
                self.basedir, self.expname, '{}_{:06d}_full.npy'.format(k, i)
            )
            
            if not isinstance(self.calib_vars[k], np.ndarray):
                var_to_save = self.calib_vars[k].numpy()
            else:
                var_to_save = self.calib_vars[k]

            np.save(calib_path, var_to_save)

            if 'pose' in k and len(var_to_save.shape) == 2:
                print(calib_path)
                print(k)
                print(var_to_save.shape)
                full_var = np.array(se3_hat(var_to_save))
                np.save(calib_path_full, full_var)

        print('Saved calibration variables')

    def load_calibration(self, i):
        # Poses
        for k in self.calib_vars:
            calib_path = os.path.join(
                self.basedir, self.expname, '{}_{:06d}.npy'.format(k, i)
            )
            
            if os.path.exists(calib_path):
                temp_var = np.load(calib_path)
                
                if not isinstance(self.calib_vars[k], tf.Variable):
                    self.calib_vars[k] = temp_var
                else:
                    self.calib_vars[k].assign(temp_var)
            else:
                print(f'Calibration variable {k} not found at {calib_path}')

        print('Loaded calibration variables')

    def save_codes(self, i):
        path = os.path.join(
            self.basedir, self.expname, 'codes_{:06d}.npy'.format(i)
        )
        np.save(path, self.temporal_codes.numpy())
        print('saved codes at', path)

    def save_dc_offset(self, i):
        path = os.path.join(
            self.basedir, self.expname, 'dc_offset_{:06d}.npy'.format(i)
        )
        np.save(path, self.dc_offset.numpy())
        print('saved dc offset at', path)

    def load_dc_offset(self, i):
        dc_offset_path =  os.path.join(
                self.basedir, self.expname, 'dc_offset_{:06d}.npy'.format(i)
            )
        
        if os.path.exists(dc_offset_path) :
            temp_var = np.load(dc_offset_path)
            self.dc_offset.assign(temp_var)
            print('Loaded DC offset')

    def save_weights(self, i):
        for k in self.models:
            path = os.path.join(
                self.basedir, self.expname, '{}_{:06d}.npy'.format(k, i)
            )

            np.save(path, self.models[k].get_weights())

        print('Saved weights')
    
    def save_all(self, i):
        self.save_calibration(i)
        self.save_weights(i)

        if self.args.dynamic and self.args.temporal_embedding == 'latent':
            self.save_codes(i)

        if self.args.use_quads:
            self.save_dc_offset(i)

    def get_training_args(self, i, render_kwargs, args):
        render_kwargs['num_views'] = args.num_views
        render_kwargs['static_scene'] = i < args.static_scene_iters
        sparsity_weight = args.sparsity_weight

        if render_kwargs['static_scene'] and args.dynamic:
            render_kwargs['network_query_fn'] = (self.all_query_fns[1],)
            render_kwargs['dynamic'] = False
            render_kwargs['render_rays_fn'] = render_rays
            sparsity_weight = 0.0
        elif args.dynamic:
            render_kwargs['network_query_fn'] = self.all_query_fns
            render_kwargs['dynamic'] = True
            render_kwargs['render_rays_fn'] = render_rays_dynamic

        render_kwargs['use_phase_calib'] = i > args.no_phase_calib_iters and args.use_phase_calib

        render_kwargs['use_phasor'] = i > args.no_phase_iters and args.use_phasor

        render_kwargs['use_variance_weighting'] = \
            args.use_variance_weighting and (i > self.args.no_variance_iters)

        render_kwargs['use_tof_uncertainty'] = \
            args.use_tof_uncertainty
         
        images_to_log = [
            'color', 'tof_cos', 'tof_sin', 'tof_amp', 'disp'
        ]
        
        if self.args.dataset_type == 'quads':
            images_to_log = [
            'color', 'disp', 'tofQuad_0', 'tofQuad_1','tofQuad_2','tofQuad_3','tof_cos','tof_amp',
            ] + (['scene_flow'] if (args.scene_flow and render_kwargs['dynamic']) else [])

        # Calculate empty weight
        if args.sparsity_weight_decay < 1.0 and args.sparsity_weight_decay_steps != 0 and not render_kwargs['static_scene']:
            decay_exp = float(i) / ((args.tof_weight_decay_steps * 1000) - args.static_scene_iters)
            sparsity_weight = np.power(args.sparsity_weight_decay, decay_exp) * sparsity_weight
        
        # Calculate tof weight
        if args.tof_weight_decay < 1.0 and args.tof_weight_decay_steps != 0:
            decay_exp = min(i // (args.tof_weight_decay_steps * 1000), 1.0)
            tof_weight = (args.tof_weight_decay ** decay_exp) * args.tof_weight
        else:
            tof_weight = args.tof_weight
        # We want to use the same weight for all the quads during pretraining
        tof_weight_2 = tof_weight

        # Calculate depth weight
        if args.depth_weight_decay < 1.0 and args.depth_weight_decay_steps != 0:
            decay_exp = np.minimum(i // (args.depth_weight_decay_steps * 1000))
            depth_weight = np.power(args.depth_weight_decay, decay_exp) * args.depth_weight
        else:
            depth_weight = args.depth_weight

        # Calculate scene flow data weight
        if (i > ((args.scene_flow_weight_decay_steps * 1000) + args.pretraining_stage1_iters)) :
            scene_flow_weight = 0
        elif i < args.pretraining_stage1_iters :
            scene_flow_weight = 0
        else :
            decay_fac = ((args.scene_flow_weight_decay_steps * 1000.0 - (i - args.pretraining_stage1_iters)) / (args.scene_flow_weight_decay_steps * 1000.0))
            scene_flow_weight = decay_fac * args.scene_flow_weight
            
        # warmup phase for the warped tof loss
        if i > args.pretraining_stage1_iters and args.warped_tof_weight > 0:
            warped_tof_weight = args.warped_tof_weight
            if (i - args.pretraining_stage1_iters) < args.warped_tof_warmup_iters and args.warped_tof_warmup_iters > 0:
                decay_fac = (i - args.pretraining_stage1_iters) / args.warped_tof_warmup_iters
                warped_tof_weight *= decay_fac
            tof_weight = min(tof_weight, warped_tof_weight)
        else:
            warped_tof_weight = 0
            if i > args.pretraining_stage1_iters:
                tof_weight = 0.0

        render_kwargs['quad_multiplier'] = args.quad_multiplier
        render_kwargs['tof_multiplier'] = args.tof_multiplier
        render_kwargs['normalize_render_weights'] = args.normalize_render_weights
        render_kwargs['use_relative_tof_loss'] = False if i < self.args.pretraining_stage2_iters else args.use_relative_tof_loss
        render_kwargs['supervise_amplitude'] = args.supervise_amplitude
        return tof_weight, args.color_weight, depth_weight, sparsity_weight, scene_flow_weight, tof_weight_2, warped_tof_weight, images_to_log

    def get_test_render_args(self, render_kwargs_train, i, args):
        render_kwargs_test = {
            k: render_kwargs_train[k] for k in render_kwargs_train
            }
        render_kwargs_test['perturb'] = False
        render_kwargs_test['raw_noise_std'] = 0.
        
        return render_kwargs_test
    
    def apply_gradients(self, loss, tape, filter_grad_vars=None):
        # Filter
        if filter_grad_vars is not None:
            grad_vars = [g for g in self.grad_vars if filter_grad_vars in g.name]
        else:
            grad_vars = self.grad_vars

        # Optimize poses and other vars separately
        if self.args.optimize_poses or self.args.optimize_phase_offset or self.args.optimize_emitter_intensity:
            # TODO (mokunev): just check that grad_vars are not empty
            gradients = tape.gradient(loss, grad_vars + self.grad_calib_vars)

            gradients_vars = gradients[:len(grad_vars)]
            grads_and_vars = [(grad, var) for (grad, var) in zip(gradients_vars, grad_vars) if grad is not None]
            self.optimizer.apply_gradients(
                    grads_and_vars
                )

            gradients_calib_vars = gradients[len(grad_vars):]
            self.calib_optimizer.apply_gradients([
                (grad, var) for (grad, var) in zip(gradients_calib_vars, self.grad_calib_vars) if grad is not None
                ]
                )
        # Optimize all vars together
        else:
            gradients = tape.gradient(loss, grad_vars)
            grads_and_vars = [(grad, var) for (grad, var) in zip(gradients, grad_vars) if grad is not None]
            self.optimizer.apply_gradients(
                    grads_and_vars
                )
        try:
            if gradients is not None:
                for grad in gradients:
                    if grad is not None:
                        tf.debugging.check_numerics(grad, message='Checking gradients')
        except Exception as e:
            print("Checking gradients : Tensor had NaN values")
        return grads_and_vars
            
    def get_ray_batch(self, coords, pose, light_pose, key):
        rays_o, rays_d = self.generate_rays[f'{key}_coords'](
            coords[..., 1], coords[..., 0], pose
        )
        
        light_pos = tf.cast(
            tf.broadcast_to(light_pose[..., :3, -1], rays_o.shape),
            tf.float32
            )
        batch_rays = tf.stack([rays_o, rays_d, light_pos], 0)

        return batch_rays

    def _train_step(
        self,
        train_iter,
        img_i,
        coords,
        batch_images, # Ground truth for img_i
        render_kwargs_train,
        outputs,
        losses,
        key,
        batch_images_partial = None, # Ground truth for partial flow img_i
        ):

        # Setup args
        render_kwargs_train = {
            k: render_kwargs_train[k] for k in render_kwargs_train
        }
        
        # Render 
        render_kwargs_train['outputs'] = outputs
        use_quads = render_kwargs_train['use_quads']

        # Storing information needed to project SF to OF
        render_kwargs_train['color_intrinsics'] = self.dataset['color_intrinsics'][0][:3, :3]
        render_kwargs_train['tof_intrinsics'] = self.dataset['tof_intrinsics'][0][:3, :3]
        render_kwargs_train['backwards_pose'] = self.color_poses[max(img_i - 4, 0)]
        render_kwargs_train['forward_pose'] = self.color_poses[min(img_i + 4, len(getattr(self, 'color_poses')) - 1)]

        loss_dict = {
            'tofQuad0' : 0.0,
            'tofQuad1' : 0.0,
            'tofQuad2' : 0.0,
            'tofQuad3' : 0.0,
            'supervised_amp': 0.0,
            'color' : 0.0,
            'scene_flow_color' : 0.0,
            'scene_flow_tof' : 0.0,
            'scene_flow_minimal_reg' : 0.0,
            'scene_flow_smoothness_temporal_reg' : 0.0,
            'scene_flow_smoothness_spatial_reg' : 0.0,
            'cycle_consistency_reg' : 0.0,
            'blend_weight_reg' : 0.0,
            'blend_weight_weak_prior_reg' : 0.0,
            'static_blending_reg' : 0.0,
            'disocclusion_reg' : 0.0,
            'warped_tofQuad0' : 0.0,
            'warped_tofQuad1' : 0.0,
            'warped_tofQuad2' : 0.0,
            'warped_tofQuad3' : 0.0,
            'warped_color' : 0.0,
            'empty_reg': 0.0,
        }

        with tf.GradientTape() as tape:
            # Get poses as matrices
            pose = getattr(self, f'{key}_poses')[img_i]
            light_pose = getattr(self, f'{key}_light_poses')[img_i]

            # Get rays
            batch_rays = self.get_ray_batch(
                coords, pose, light_pose, key
            )

            psnr = None
            psnr0 = None
            
            loss = 0.0
            img_loss, img_loss0 = 0.0, 0.0
            metrics = dict()

            # Make predictions for color, disparity, accumulated opacity.
            if use_quads:
                if self.args.dataset_type == 'quads': 
                    fractional_img_i = img_i / 4.0  
                    
                    scene_flow_raw_map = None # Minimize the number of queries for raw scene flow @ time i
                    # Compute warping based losses
                    for warp_offset in render_kwargs_train["warp_offsets"] :
                        if ((img_i + 4 * warp_offset) < self.args.view_start) or ((img_i + 4 * warp_offset) >= len(getattr(self, f'{key}_poses'))) :
                            continue
                        if warp_offset != 0 and (not render_kwargs_train['dynamic']):
                            continue

                        img_loss, img_loss0 = 0.0, 0.0
                        render_kwargs_train["warp_offset"] = warp_offset 
                    
                        outputs = render(
                            H=self.image_sizes[key][1], W=self.image_sizes[key][0],
                            chunk=self.args.chunk, rays=batch_rays,
                            image_index=fractional_img_i,
                            **render_kwargs_train
                        )

                        

                        if (warp_offset == 0) :
                            if render_kwargs_train['dynamic']:
                                scene_flow_raw_map = outputs['scene_flow_raw_map']
                                if self.args.disocclusion :
                                    disocclusion_map = outputs['disocclusion_map']
                                    disocclusion_raw_map = outputs['disocclusion_raw_map']

                        for loss_key in losses:
                            _img_loss = _img_loss0 = 0
                            if self.loss_weights[loss_key] > 0 :
                                if loss_key == 'tof' :
                                    if (warp_offset == 0) :
                                        if self.args.train_on_phasor:
                                            quads = []
                                            quads.append(batch_images[f'{loss_key}Quad0_images'][..., 0])

                                            # TODO: Linear Decay Prior on non-gt quads
                                            # Technically this can be applied for all quads, but there's more error for later quads
                                            for quad in range(1, self.args.n_quads_optimize):
                                                partial_img_i = quad + 2
                                                quads.append(batch_images_partial[f'tofQuad{quad}_images'][partial_img_i][..., 0])
                                            perm = self.dataset['tof_permutation']
                                            quads = tf.gather(quads, perm)
                                            if self.args.phasor_type == 'pre_depth':
                                                phasor_cos_gt = quads[0] - quads[1]
                                                phasor_sin_gt = quads[2] - quads[3]
                                            elif self.args.phasor_type == 'post_depth':
                                                amp = get_amplitude(quads[0] - quads[1], quads[2] - quads[3])
                                                phase = get_phase(quads[0] - quads[1], quads[2] - quads[3])
                                                phasor_cos_gt, phasor_sin_gt = get_phasor(phase, amp)
                                            else:
                                                raise NotImplementedError

                                            cos_mse = img2mse(phasor_cos_gt, outputs['tof_map'][..., 0]) * self.loss_weights[loss_key]
                                            _img_loss += cos_mse
                                            sin_mse = img2mse(phasor_sin_gt, outputs['tof_map'][..., 1]) * self.loss_weights[loss_key]
                                            _img_loss += sin_mse

                                            psnr_cos = mse2psnr(cos_mse).numpy()
                                            psnr_sin = mse2psnr(sin_mse).numpy()
                                            metrics['psnr_cos'] = psnr_cos
                                            metrics['psnr_sin'] = psnr_sin
                                        else:
                                            mode = 'relative' if render_kwargs_train['use_relative_tof_loss'] else 'absolute'
                                            quads = []
                                            for quad in range(self.args.n_quads_optimize):
                                                _img_loss_quad, _img_loss0_quad = self.img_loss_fns[loss_key](
                                                    batch_images[f'{loss_key}Quad{quad}_images'], outputs, self.loss_weights[loss_key], 
                                                    quad, use_mask = True, mode=mode, norm=self.args.tof_loss_norm
                                                )
                                                loss_dict[f'{loss_key}Quad{quad}'] += _img_loss_quad
                                                _img_loss += _img_loss_quad
                                                _img_loss0 += _img_loss0_quad
                                                if _img_loss_quad != 0.0:
                                                    quads.append(batch_images[f'{loss_key}Quad{quad}_images'][..., 0])
                                                    if self.args.tof_loss_norm == 'L1':
                                                        mse, mse0 = self.img_loss_fns[loss_key](
                                                            batch_images[f'{loss_key}Quad{quad}_images'], outputs, self.loss_weights[loss_key], 
                                                            quad, use_mask = True, mode=mode, norm='L2'
                                                        )
                                                    elif self.args.tof_loss_norm == 'L2':
                                                        mse, mse0 = _img_loss_quad, _img_loss0_quad
                                                    mse /= self.loss_weights[loss_key]
                                                    mse0 /= self.loss_weights[loss_key]
                                                    psnr = mse2psnr(mse).numpy()
                                                    psnr0 = mse2psnr(mse0).numpy()
                                                    metrics[f'Quad_{quad}_psnr'] = psnr
                                                    metrics[f'Quad_{quad}_psnr0'] = psnr0

                                            # TODO: Linear Decay Prior on non-gt quads
                                            # Technically this can be applied for all quads, but there's more error for later quads
                                            if self.loss_weights['tof_2'] > 0:
                                                if self.args.partial_scene_flow :
                                                    for quad in range(1, self.args.n_quads_optimize):
                                                        partial_img_i = quad + 2
                                                        _img_loss_quad, _img_loss0_quad = self.img_loss_fns[loss_key](
                                                            batch_images_partial[f'tofQuad{quad}_images'][partial_img_i], outputs, self.loss_weights['tof_2'], 
                                                            quad, use_mask = True, mode=mode, norm=self.args.tof_loss_norm
                                                        )
                                                        loss_dict[f'{loss_key}Quad{quad}'] += _img_loss_quad
                                                        _img_loss += _img_loss_quad
                                                        _img_loss0 += _img_loss0_quad
                                                        if _img_loss_quad != 0.0:
                                                            quads.append(batch_images_partial[f'tofQuad{quad}_images'][partial_img_i][..., 0])
                                                            if self.args.tof_loss_norm == 'L1':
                                                                mse, mse0 = self.img_loss_fns[loss_key](
                                                                    batch_images_partial[f'tofQuad{quad}_images'][partial_img_i], outputs, self.loss_weights['tof_2'], 
                                                                    quad, use_mask = True, mode=mode, norm='L2'
                                                                )
                                                            elif self.args.tof_loss_norm == 'L2':
                                                                mse, mse0 = _img_loss_quad, _img_loss0_quad
                                                            mse /= self.loss_weights['tof_2']
                                                            mse0 /= self.loss_weights['tof_2']
                                                            psnr = mse2psnr(mse).numpy()
                                                            psnr0 = mse2psnr(mse0).numpy()
                                                            metrics[f'Quad_{quad}_pretrain_psnr'] = psnr
                                                            metrics[f'Quad_{quad}_pretrain_psnr0'] = psnr0
                                                
                                                assert len(quads) == 4, f"Expected 4 quads, got {len(quads)}"

                                                if render_kwargs_train['supervise_amplitude'] == 'derived':
                                                    _img_loss, _img_loss0 = self.img_loss_fns['amp_derived_loss'](
                                                        outputs, self.loss_weights[loss_key]
                                                    )
                                                    loss_dict['supervised_amp'] += _img_loss
                                                    _img_loss += _img_loss
                                                    _img_loss0 += _img_loss0
                                                elif render_kwargs_train['supervise_amplitude'] == 'gt':
                                                    quads = tf.gather(quads, self.dataset['tof_permutation'])
                                                    amp_gt = get_amplitude(quads[0] - quads[1], quads[2] - quads[3])
                                                    _img_loss, _img_loss0 = self.img_loss_fns['amp_gt_loss'](
                                                        outputs, amp_gt, self.loss_weights[loss_key]
                                                    )
                                                    loss_dict['supervised_amp'] += _img_loss
                                                    _img_loss += _img_loss
                                                    _img_loss0 += _img_loss0
                                else: 
                                    # Only compute scene flow data prior loss for warp_offset_0
                                    if (loss_key == 'scene_flow_color' and render_kwargs_train['dynamic']):
                                        # Dataloader loads y, x coordinates
                                        # RAFT + Predicted Scene Flow is x, y
                                        source_coordinates_x = coords[:, 1:]
                                        source_coordinates_y = coords[:, :1]
                                        source_coordinates = tf.concat([source_coordinates_x, source_coordinates_y], 1)
                                        source_coordinates = tf.cast(source_coordinates, dtype = tf.float32)

                                        if (warp_offset == 0) :
                                            if (img_i + 4) < len(getattr(self, 'color_poses')) :
                                                __img_loss, __img_loss0 = self.img_loss_fns[loss_key](
                                                    batch_images['forward_flow'], outputs, 1.0, outputs_key = 'forward_proj_map', source_coordinates = source_coordinates
                                                )

                                                _img_loss += __img_loss * self.loss_weights[loss_key]
                                                _img_loss0 += __img_loss0 * self.loss_weights[loss_key]
                                                loss_dict[loss_key] += __img_loss

                                                forward_metrics = compute_flow_metrics(
                                                    source_coordinates, batch_images['forward_flow'], outputs['forward_proj_map']
                                                )
                                                metrics['forward_epe'] = forward_metrics['epe']
                                                metrics['forward_dynamic_epe'] = forward_metrics['dynamic_epe']
                                                metrics['forward_dynamic_relative_error'] = forward_metrics['dynamic_relative_error']

                                            if (img_i - 4) >= self.args.view_start :
                                                __img_loss, __img_loss0 = self.img_loss_fns[loss_key](
                                                    batch_images['backward_flow'], outputs, 1.0, outputs_key = 'backwards_proj_map', source_coordinates = source_coordinates
                                                )

                                                _img_loss += __img_loss * self.loss_weights[loss_key]
                                                _img_loss0 += __img_loss0 * self.loss_weights[loss_key]
                                                loss_dict[loss_key] += __img_loss

                                                backward_flow_metrics = compute_flow_metrics(
                                                    source_coordinates, batch_images['backward_flow'], outputs['backwards_proj_map']
                                                )

                                                metrics['backward_epe'] = backward_flow_metrics['epe']
                                                metrics['backward_dynamic_epe'] = backward_flow_metrics['dynamic_epe']
                                                metrics['backward_dynamic_relative_error'] = backward_flow_metrics['dynamic_relative_error']


                                    elif (loss_key == 'color') :
                                        if (warp_offset == 0) :
                                            _img_loss, _img_loss0 = self.img_loss_fns[loss_key](
                                                batch_images[f'{loss_key}_images'], outputs, self.loss_weights[loss_key], use_mask = True
                                            )
                                            loss_dict[loss_key] += _img_loss

                                            psnr = mse2psnr(_img_loss / self.loss_weights[loss_key]).numpy()
                                            psnr0 = mse2psnr(_img_loss0 / self.loss_weights[loss_key]).numpy()
                                            metrics['color_psnr'] = psnr
                                            metrics['color_psnr0'] = psnr0
                                        else :
                                            if self.loss_weights['warped_color'] > 0:
                                                occ_map_idx = None
                                                if warp_offset == 1 :
                                                    occ_map_idx = 7
                                                elif warp_offset == -1 :
                                                    occ_map_idx = 0
                                                _img_loss, _img_loss0 = self.img_loss_fns[loss_key](
                                                    batch_images[f'{loss_key}_images'], outputs, self.loss_weights['warped_' + loss_key], use_mask = True, 
                                                    occ_maps = disocclusion_map if self.args.disocclusion else None,   
                                                    occ_map_idx = occ_map_idx if self.args.disocclusion else None
                                                )
                                                loss_dict['warped_' + loss_key] += _img_loss 
                                                psnr = mse2psnr(_img_loss / self.loss_weights['warped_color']).numpy()
                                                psnr0 = mse2psnr(_img_loss0 / self.loss_weights['warped_color']).numpy()
                                                metrics['warped_color_psnr'] = psnr
                                                metrics['warped_color_psnr0'] = psnr0

                            img_loss += _img_loss
                            img_loss0 += _img_loss0
                        # Total
                        loss += img_loss + img_loss0
                    
                        # Regularization losses
                        if warp_offset == 0 :
                            for loss_key in self.reg_loss_fns:     
                                if self.loss_weights[loss_key] > 0:
                                    if loss_key == 'cycle_consistency' and render_kwargs_train['dynamic']:
                                        reg_loss = cycle_consistency_loss(outputs, img_i, self.args.num_frames,
                                                                          bw_disocclusion = (disocclusion_raw_map[..., 0] if self.args.disocclusion else None),
                                                                            fw_disocclusion = (disocclusion_raw_map[..., 7] if self.args.disocclusion else None),
                                                                            start_idx = self.args.view_start)

                                    elif loss_key == 'blend_weight_weak_prior' and render_kwargs_train['dynamic']:
                                        reg_loss = blending_weight_weak_prior_loss(outputs, img_i, self.args.num_frames, self.args.blend_weight_weak_prior)
                                    else :
                                        if (not render_kwargs_train['dynamic']):
                                            continue
                                        reg_loss = self.reg_loss_fns[loss_key](outputs, img_i, self.args.num_frames, start_idx = self.args.view_start) 
                                    loss_dict[f'{loss_key}_reg'] += reg_loss
                                    loss += reg_loss * self.loss_weights[loss_key]
                            # if img_i < 100:
                            #      loss += tf.reduce_mean(tf.abs(outputs['phase_amp_raw'] - 0.5))
                    
                    if (not self.args.train_on_phasor) and self.args.partial_scene_flow and render_kwargs_train['dynamic'] and ('scene_flow_tof' in losses):
                        # TODO: [Possible Fix] Raw scene flow should actually be from neighboring time moments
                        # TODO: [Optional] Under cycle consistency, can use flow @ time t, and just cleverly negate stuff

                        # Reset warp_offset to 0
                        
                        render_kwargs_train["warp_offset"] = 0

                        # Forward flow should be forward flow of t-1 -> t
                        forward_outputs = render(
                            H=self.image_sizes[key][1], W=self.image_sizes[key][0],
                            chunk=self.args.chunk, rays=batch_rays,
                            image_index=((img_i - 4)/4.0),
                            **render_kwargs_train
                        )

                        forward_flow = forward_outputs['scene_flow_raw_map'][..., 0:3]

                        # Backward flow should be backward flow of t+1 -> t
                        backward_outputs = render(
                            H=self.image_sizes[key][1], W=self.image_sizes[key][0],
                            chunk=self.args.chunk, rays=batch_rays,
                            image_index=((img_i + 4)/4.0),
                            **render_kwargs_train
                        )

                        backward_flow = backward_outputs['scene_flow_raw_map'][..., 3:]

                        scene_flow_raw_map = tf.concat((forward_flow, backward_flow), 2)

                        partial_disocclusion_map = None
                        if self.args.disocclusion :
                            # TODO: figure this out
                            # TODO: should masks come from neighbors or t

                            # occ_mask correspond to [-1 ... -0.5 ... +0.5 ... +1]

                            # forward_outputs are warping from (t-0.5) -> t
                            # so we want the forward flow disocclusion masks (last 4 dimensions, grab first 3/4 for partial)
                            forward_dis = forward_outputs['disocclusion_map'][..., 4:7]

                            # backward_outputs are warping from (t+0.5) -> t
                            # so we want the backward flow disocclusion masks (first 4 dimensions, grab last 3/4 for partial)
                            backward_dis = backward_outputs['disocclusion_map'][..., 1:4]

                            partial_disocclusion_map = tf.concat((forward_dis, backward_dis), 1)                                                
                        #############################################
                        
                        # Partial scene flow losses
                        img_loss = img_loss0 = 0.0
                        partial_flow_offset_i = [-3, -2, -1, 1, 2, 3]
            
                        for partial_img_i, partial_offset in enumerate(partial_flow_offset_i) :
                            if ((img_i + partial_offset) < self.args.view_start) or ((img_i + partial_offset) > len(getattr(self, f'{key}_poses')) - 4) : # Bounds check
                                continue
                            
                            partial_pose = self.tof_poses[img_i + partial_offset]
                            partial_light_pose = self.tof_light_poses[img_i + partial_offset]

                            # Get rays
                            partial_batch_rays = self.get_ray_batch(
                                coords, partial_pose, partial_light_pose, key
                            )
                            partial_fractional_img_i = fractional_img_i + (partial_offset / 4.0)
                            # render_kwargs_train["warp_offset"] =  (partial_offset / 4.0)

                            # Supervise w/ partial_fractional_img_i with warped ray rendered @ fractional_img_i
                            render_kwargs_train["warp_offset"] = fractional_img_i - partial_fractional_img_i

                            outputs = render(
                                H=self.image_sizes[key][1], W=self.image_sizes[key][0],
                                chunk=self.args.chunk, rays=partial_batch_rays,
                                image_index=partial_fractional_img_i,
                                scene_flow_raw_map = scene_flow_raw_map,
                                **render_kwargs_train
                            )

                            _img_loss = _img_loss0 = 0

                            loss_key = 'warped_tof'
                            if self.loss_weights[loss_key] > 0 : 
                                mode = 'relative' if render_kwargs_train['use_relative_tof_loss'] else 'absolute'
                                for quad in range(self.args.n_quads_optimize):
                                        _img_loss_quad, _img_loss0_quad = self.img_loss_fns['tof'](
                                            batch_images_partial[f'tofQuad{quad}_images'][partial_img_i], outputs, self.loss_weights[loss_key], 
                                            quad, use_mask = True, mode=mode, occ_maps = partial_disocclusion_map, 
                                            occ_map_idx = partial_img_i if self.args.disocclusion else None,
                                            norm=self.args.tof_loss_norm
                                        )
                                        loss_dict[f'{loss_key}Quad{quad}'] += _img_loss_quad
                                        _img_loss += _img_loss_quad
                                        _img_loss0 += _img_loss0_quad

                                        if _img_loss_quad != 0.0:
                                            if self.args.tof_loss_norm == 'L1':
                                                mse, mse0 = self.img_loss_fns['tof'](
                                                    batch_images_partial[f'tofQuad{quad}_images'][partial_img_i], outputs, self.loss_weights[loss_key], 
                                                    quad, use_mask = True, mode=mode, occ_maps = partial_disocclusion_map, 
                                                    occ_map_idx = partial_img_i if self.args.disocclusion else None,
                                                    norm='L2'
                                                )
                                            elif self.args.tof_loss_norm == 'L2':
                                                mse, mse0 = _img_loss_quad, _img_loss0_quad
                                            mse /= self.loss_weights[loss_key]
                                            mse0 /= self.loss_weights[loss_key]
                                            psnr = mse2psnr(mse).numpy()
                                            psnr0 = mse2psnr(mse0).numpy()
                                            metrics[f'Quad_{quad}_warped_psnr'] = psnr
                                            metrics[f'Quad_{quad}_warped_psnr0'] = psnr0

                            img_loss += _img_loss
                            img_loss0 += _img_loss0
                        # Total
                        loss += img_loss + img_loss0

                    loss_dict["loss"] = loss
                    
                else : 
                    # TODO: [Marc] Remove/Deprecate this after confirming quadDataset + quads data works as expected
                    # Keeping this atm for backwards compat with Mitsuba Dataset + modified data prep with the quads 
                    for quad in range(4):
                        outputs = render(
                            H=self.image_sizes[key][1], W=self.image_sizes[key][0],
                            chunk=self.args.chunk, rays=batch_rays,
                            image_index=img_i + quad / 4.0,
                            **render_kwargs_train
                        )
                        # Image losses
                        for loss_key in losses:
                            if self.loss_weights[loss_key] > 0:
                                if loss_key == 'tof':
                                    _img_loss, _img_loss0 = self.img_loss_fns[loss_key](
                                        batch_images[f'{loss_key}_images'], outputs, self.loss_weights[loss_key], quad
                                    )

                                else: 
                                    if quad == 0:
                                        _img_loss, _img_loss0 = self.img_loss_fns[loss_key](
                                            batch_images[f'{loss_key}_images'], outputs, self.loss_weights[loss_key]
                                        )

                                        if loss_key == 'color':
                                            psnr = mse2psnr(_img_loss / self.loss_weights[loss_key]).numpy()
                                            psnr0 = mse2psnr(_img_loss0 / self.loss_weights[loss_key]).numpy()
                                            metrics['psnr'] = psnr
                                            metrics['psnr0'] = psnr0
                            img_loss += _img_loss
                            img_loss0 += _img_loss0
                        # Regularization losses

                        for loss_key in self.reg_loss_fns:
                            if self.loss_weights[loss_key] > 0:
                                reg_loss = self.reg_loss_fns[loss_key](outputs) * self.loss_weights[loss_key]
                                loss += reg_loss

                        # Total
                    loss += img_loss + img_loss0
                    loss_dict["loss"] = loss
            else:
                # Makes render_rays_dynamic compatible w/ ToRF Baseline
                render_kwargs_train["warp_offset"] = 0
                
                # Baseline ToRF
                outputs = render(
                    H=self.image_sizes[key][1], W=self.image_sizes[key][0],
                    chunk=self.args.chunk, rays=batch_rays,
                    image_index=img_i,
                    **render_kwargs_train
                )

                # Image losses
                img_loss, img_loss0 = 0.0, 0.0
                psnr = 0.0
                psnr0 = 0.0

                for loss_key in losses:
                    if self.loss_weights[loss_key] > 0:
                        _img_loss, _img_loss0 = self.img_loss_fns[loss_key](
                            batch_images[f'{loss_key}_images'], outputs, self.loss_weights[loss_key]
                        )
                        img_loss += _img_loss
                        img_loss0 += _img_loss0

                        if loss_key == 'color':
                            psnr = mse2psnr(img_loss / self.loss_weights[loss_key]).numpy()
                            psnr0 = mse2psnr(img_loss0 / self.loss_weights[loss_key]).numpy()
                            metrics['psnr'] = psnr
                            metrics['psnr0'] = psnr0
                
                # Regularization losses
                loss = 0.0

                for loss_key in self.reg_loss_fns:
                    if self.loss_weights[loss_key] > 0:
                        reg_loss = self.reg_loss_fns[loss_key](outputs) * self.loss_weights[loss_key]
                        loss += reg_loss

                # Total
                loss += img_loss + img_loss0
                loss_dict["loss"] = loss
        # Gradients
        # Check if loss was never computed and only apply the gradients in case it was
        if loss == 0.0:
            gradients = None
        else:
            gradients = self.apply_gradients(loss, tape)

        return loss_dict, metrics, gradients
    
    def train_step(
        self,
        i,
        render_kwargs_train,
        args
        ):

        loss = 0.0

        tof_batch_key = ['tof_images']
        if self.args.dataset_type == 'quads':
            tof_batch_key = ['tofQuad0_images', 'tofQuad1_images', 'tofQuad2_images', 'tofQuad3_images']            

        if not args.train_both:
            # TOF
            if (i < args.no_color_iters or (i % 2) == 1):
                key = 'tof'
                batch_outputs = tof_batch_key + (['depth_images'] if args.use_depth_loss else []) + (['forward_flow', 'backward_flow'] if (args.scene_flow and render_kwargs_train['dynamic']) else [])
                outputs = ['tof_map', 'acc_map'] + (['depth_map'] if args.use_depth_loss else []) +  (['scene_flow_map', 'depth_map', 'forward_proj_map', 'backwards_proj_map'] if (args.scene_flow and render_kwargs_train['dynamic']) else []) +  (['disocclusion_map', 'disocclusion_raw_map'] if (args.disocclusion and render_kwargs_train['dynamic']) else [])
                losses = ['tof'] + (['depth'] if args.use_depth_loss else []) + (['scene_flow_tof'] if (args.scene_flow and render_kwargs_train['dynamic']) else [])
            # Color
            else:
                key = 'color'
                batch_outputs = ['color_images'] + (['depth_images'] if args.use_depth_loss else []) + (['forward_flow', 'backward_flow'] if (args.scene_flow and render_kwargs_train['dynamic']) else [])
                outputs = ['color_map', 'acc_map'] + (['depth_map'] if args.use_depth_loss else []) +  (['scene_flow_map', 'depth_map', 'forward_proj_map', 'backwards_proj_map'] if (args.scene_flow and render_kwargs_train['dynamic']) else []) +  (['disocclusion_map', 'disocclusion_raw_map'] if (args.disocclusion and render_kwargs_train['dynamic']) else [])
                losses = ['color'] + (['depth'] if args.use_depth_loss else []) + (['scene_flow_color'] if (args.scene_flow and render_kwargs_train['dynamic']) else [])
        else:
            # Both
            key = 'tof'
            batch_outputs = tof_batch_key +  ['color_images'] + (['depth_images'] if args.use_depth_loss else []) + (['forward_flow', 'backward_flow'] if (args.scene_flow and render_kwargs_train['dynamic']) else [])
            outputs = ['tof_map', 'color_map', 'acc_map'] + (['depth_map'] if args.use_depth_loss else []) +  (['scene_flow_map', 'depth_map', 'forward_proj_map', 'backwards_proj_map'] if (args.scene_flow and render_kwargs_train['dynamic']) else []) +  (['disocclusion_map', 'disocclusion_raw_map'] if (args.disocclusion and render_kwargs_train['dynamic']) else [])
            losses = ['tof', 'color'] + (['depth'] if args.use_depth_loss else []) + (['scene_flow_color', 'scene_flow_tof'] if (args.scene_flow and render_kwargs_train['dynamic']) else [])

        # Get batch
        img_i, coords, batch_images = self.dataloader.get_batch(
            self.dataloader.i_train,
            args.N_rand,
            self.image_sizes[key][1], self.image_sizes[key][0],
            args,
            outputs=batch_outputs
            )
        
        # Get partial image batch to supervise partial flow
        batch_images_partial = None
        if args.partial_scene_flow :
            offset_i = [-3, -2, -1, 1, 2, 3]
            i_list = [(x + img_i) for x in offset_i]
            i_list = [max(0, min(y, args.num_views)) for y in i_list] # clip
            batch_images_partial = self.dataloader.get_batch_multiple(i_list, coords, outputs = ['tofQuad0_images', 'tofQuad1_images', 'tofQuad2_images', 'tofQuad3_images'])

        # Train step
        loss, metrics, gradients = self._train_step(
            i,
            img_i,
            coords,
            batch_images,
            render_kwargs_train,
            outputs,
            losses,
            key,
            batch_images_partial = batch_images_partial
        )

        if (i % 1000) <= 1:
            print("Printing loss values")
            print(loss)
            print("Printing metrics")
            print(metrics)

        # Copy poses for intermediate frames
        # for idx in range(len(self.color_poses) // 4):
        #     for k in range(3):
        #         if isinstance(self.poses['color_poses'], tf.Variable):
        #             self.poses['color_poses'][idx * 4 + (k + 1)].assign(self.poses['color_poses'][idx * 4])
        #         if isinstance(self.poses['tof_poses'], tf.Variable):
        #             self.poses['tof_poses'][idx * 4 + (k + 1)].assign(self.poses['tof_poses'][idx * 4])
        return loss, metrics, gradients
    
    def setup_dataset(self):
        ## Load data
        if self.args.dataset_type == 'real':
            dataset = RealDataset(self.args)
        elif self.args.dataset_type == 'ios':
            dataset = IOSDataset(self.args)
        elif self.args.dataset_type == 'mitsuba':
            dataset = MitsubaDataset(self.args)
        elif self.args.dataset_type == 'quads':
            dataset = QuadsDataset(self.args)
        else:
            dataset = ToFDataset(self.args)

        self.dataset = dataset.dataset
        self.dataloader = dataset

        if self.args.dataset_type == 'quads':
            print(
            'Loaded dataset',
            self.args.dataset_type,
            self.dataset['tofQuad0_images'].shape,
            self.dataset['tof_intrinsics'][0],
            self.dataset['color_intrinsics'][0],
            self.args.datadir
        )
        else :
            print(
                'Loaded dataset',
                self.args.dataset_type,
                self.dataset['tof_images'].shape,
                self.dataset['tof_intrinsics'][0],
                self.dataset['color_intrinsics'][0],
                self.args.datadir
            )

        
        ## Bounds
        self.near = tf.reduce_min(self.dataset['bounds']) * 0.9
        self.far = tf.reduce_max(self.dataset['bounds']) * 1.1
        self.min_vis_depth = self.dataset['bounds'].min()

        print('NEAR FAR', self.near, self.far)

        ## Show images
        if self.args.show_images:
            if self.args.dataset_type == 'quads': 
                raise NotImplementedError("Image preview not implemented for QuadsDataset")
            
            for i in range(self.args.num_views):
                plt.imshow(self.dataset['color_images'][i])
                plt.show()

                plt.imshow(self.dataset['tof_images'][i][..., 0])
                plt.show()

                plt.imshow(self.dataset['tof_images'][i][..., 1])
                plt.show()

                plt.subplot(1, 2, 1)
                plt.imshow(self.dataset['tof_depth_images'][i], vmin=0, vmax=9)

                plt.subplot(1, 2, 2)
                plt.imshow(self.dataset['depth_images'][i], vmin=0, vmax=9)
                plt.show()
    
    def setup_loggers(self):
        self.basedir = self.args.basedir
        self.expname = self.args.expname

        # Logging directories
        os.makedirs(os.path.join(self.basedir, self.expname), exist_ok=True)
        f = os.path.join(self.basedir, self.expname, 'args.txt')

        with open(f, 'w') as file:
            for arg in sorted(vars(self.args)):
                attr = getattr(self.args, arg)
                file.write('{} = {}\n'.format(arg, attr))

        if self.args.config is not None:
            f = os.path.join(self.basedir, self.expname, 'config.txt')

            with open(f, 'w') as file:
                file.write(open(self.args.config, 'r').read())

        # Summary writer
        self.writer = tf.summary.create_file_writer(
            os.path.join(self.basedir, 'summaries', self.expname)
            )
        self.writer.set_as_default()

    def setup_models(self):
        ## Create model
        self.render_kwargs_train, self.render_kwargs_test, \
            self.start, self.grad_vars, self.models, self.temporal_codes = \
            create_nerf(self.args)
        self.all_query_fns = self.render_kwargs_train['network_query_fn']

        print(self.models)
        
        ## Create bounds
        bds_dict = {
            'near': tf.cast(self.near, tf.float32),
            'far': tf.cast(self.far, tf.float32)
        }

        self.render_kwargs_train.update(bds_dict)
        self.render_kwargs_test.update(bds_dict)

        ## Calibration variables
        self.setup_calibration()
        self.load_calibration(self.start - 1)

        # Load DC Offset
        if self.args.use_quads :
            self.load_dc_offset(self.start - 1)

        ## Optimizers
        self.setup_optimizers()

        ## Step
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.global_step.assign(self.start)
    
    def setup_optimizers(self):
        ## Optimizer
        if self.args.lrate_decay > 0:
            lrate = tf.keras.optimizers.schedules.ExponentialDecay(
                self.args.lrate,
                decay_steps=self.args.lrate_decay * 1000,
                decay_rate=0.1
                )
        else:
            lrate = self.args.lrate

        if self.args.optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(lrate, clipvalue=self.args.clip_grad)
        elif self.args.optimizer == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(lrate, clipvalue=self.args.clip_grad)
        else:
            raise NotImplementedError(self.args.optimizer)

        ## Calib optimizer
        if self.args.lrate_decay_calib > 0:
            lrate_calib = tf.keras.optimizers.schedules.ExponentialDecay(
                self.args.lrate_calib,
                decay_steps=self.args.lrate_decay_calib * 1000,
                decay_rate=0.1
                )
        else:
            lrate_calib = self.args.lrate_calib

        if self.args.optimizer == 'adam':
            self.calib_optimizer = tf.keras.optimizers.Adam(lrate_calib, clipvalue=self.args.clip_grad)
        elif self.args.optimizer == 'sgd':
            self.calib_optimizer = tf.keras.optimizers.SGD(lrate_calib, clipvalue=self.args.clip_grad)
        else:
            raise NotImplementedError(self.args.optimizer)

        ## Add to models
        self.models['optimizer'] = self.optimizer
        self.models['calib_optimizer'] = self.calib_optimizer

    def setup_losses(self):
        self.img_loss_fns = {
            'color': color_loss_default,
            'tof': tof_loss_default,
            'amp_derived_loss': amp_derived_loss,
            'amp_gt_loss': amp_gt_loss,
            'depth': depth_loss_default,
            'scene_flow_color': scene_flow_loss_default,
            'scene_flow_tof': scene_flow_loss_default,
        }
        self.reg_loss_fns = {
            'empty': empty_space_loss,
            'tof_poses': make_pose_loss(self, 'tof_poses'),
            'color_poses': make_pose_loss(self, 'color_poses'),
        }
        if self.args.scene_flow :
            self.reg_loss_fns['scene_flow_minimal'] = minimal_scene_flow_loss
            self.reg_loss_fns['scene_flow_smoothness_temporal'] = temporal_scene_flow_smoothness_loss
            self.reg_loss_fns['scene_flow_smoothness_spatial'] = spatial_scene_flow_smoothness_loss
            self.reg_loss_fns['cycle_consistency'] = cycle_consistency_loss
        if self.args.blend_reg :
            self.reg_loss_fns['blend_weight'] = blend_weight_entropy_loss
            self.reg_loss_fns['blend_weight_weak_prior'] = blending_weight_weak_prior_loss
            self.reg_loss_fns['static_blending'] = static_blend_weight_loss
        if self.args.disocclusion :
            self.reg_loss_fns['disocclusion'] = minimal_disocclusion_loss
        
    def update_args(self, i):
        # Get training params for current iter
        tof_weight, color_weight, depth_weight, sparsity_weight, scene_flow_weight, tof_weight_2, warped_tof_weight, self.images_to_log = \
            self.get_training_args(i, self.render_kwargs_train, self.args)

        self.loss_weights = {
            'color': color_weight,
            'tof': tof_weight,
            'tof_2' : tof_weight_2,
            'warped_color' : self.args.warped_color_weight,
            'warped_tof' : warped_tof_weight,
            'depth': depth_weight,
            'empty': sparsity_weight,
            "scene_flow_color": scene_flow_weight,
            "scene_flow_tof": scene_flow_weight,
            'tof_poses': self.args.pose_reg_weight,
            'color_poses': self.args.pose_reg_weight,
            'scene_flow_minimal' : self.args.scene_flow_minimal,
            'scene_flow_smoothness_temporal' : self.args.scene_flow_smoothness_temporal,
            'scene_flow_smoothness_spatial' : self.args.scene_flow_smoothness_spatial,
            'cycle_consistency' : self.args.cycle_consistency,
            'blend_weight' : self.args.blend_weight,
            'blend_weight_weak_prior' : self.args.blend_weight_weak_prior_reg_weight,
            'static_blending' : self.args.static_blending_loss_weight,
            'disocclusion' : self.args.disocclusion_reg_weight
        }

        # Turn off tof warping until iteration 100000
        # TODO: [Marc] make the 100,000 more generic and controlled by config
        
        if self.args.pretraining_scheme == "old" : # Used in thesis
            if i <= 100000 :
                self.loss_weights['warped_tof'] = 0
        elif self.args.pretraining_scheme == "stages" : # New 3 stage training (geometry, color flow, quad flow)
            
            # These are assuming 50k iterations total
            stage_1_iters = self.args.pretraining_stage1_iters
            stage_2_iters = self.args.pretraining_stage2_iters

            if i < stage_1_iters : # STAGE 1 (geometry only, semi-equivalent to torf)
                self.loss_weights['warped_color'] = self.loss_weights['warped_tof'] = self.loss_weights["scene_flow"] = self.loss_weights["scene_flow_color"] = self.loss_weights["scene_flow_tof"] = 0
                self.loss_weights['scene_flow_minimal'] = self.loss_weights['scene_flow_smoothness_temporal'] = self.loss_weights['scene_flow_smoothness_spatial'] = self.loss_weights['cycle_consistency'] = 0
                self.loss_weights['disocclusion'] = 0
            elif (i > stage_1_iters) and (i <= stage_2_iters) : # STAGE 2 (start optimizing color flow)
                self.loss_weights['tof_2'] = 0
                # Stopping pose optimization
                self.grad_calib_vars = []

            else : # STAGE 3 (finetuning, to be implemented, for the moment equals to stage two)
                self.loss_weights['tof_2'] = 0
                pass

        else :
            raise NotImplementedError(f"pretraining scheme {self.rgs.pretraining_scheme} NOT supported. ")


        if (i % 2500) == 0 :
            print("Printing loss weights for debugging")
            print(self.loss_weights)

        self.render_kwargs_train['emitter_intensity'] = self.calib_vars['emitter_intensity']
        self.render_kwargs_train['phase_offset'] = self.calib_vars['phase_offset']
        self.render_kwargs_train['dc_offset'] = self.dc_offset
        self.render_kwargs_train['tof_permutation'] = self.dataset['tof_inverse_permutation']
        self.render_kwargs_test = self.get_test_render_args(
            self.render_kwargs_train, i, self.args
            )

    def model_reset(self):
        self.model_reset_done = True

        for model_name in self.models:
            if 'optimizer' not in model_name:
                load_model(self.models, model_name, 0, self.args)

        if self.args.dynamic and self.args.temporal_embedding == 'latent':
            self.temporal_codes.assign(load_codes(self.temporal_codes, 0, self.args))
        
        self.setup_optimizers()
    
    def finish_calibration_pretraining(self):
        self.calibration_pretraining_done = True

        # Reset optimization parameters
        self.args.optimize_poses = True

        ## Relative pose
        if self.args.use_relative_poses \
            and not self.args.collocated_pose \
                and not self.args.optimize_relative_pose:

            self.args.optimize_relative_pose = True
            self.set_trainable_pose('relative_pose')
            self.set_saveable_pose('relative_pose')

        ## Optimizers
        self.args.lrate_calib *= self.args.lrate_calib_fac
        self.setup_optimizers()
    
    def eval(self):
        i_val = self.dataloader.i_train
        split_pose = np.array([self.tof_poses[i] for i in i_val])

        split_colors = self.dataset['color_images'][i_val]
        split_tofs = self.dataset['tof_images'][i_val]
        split_depths = self.dataset['depth_images'][i_val]
        #split_tof_depths = self.dataset['tof_depth_images'][i_val]
        split_frame_numbers = self.dataloader.i_train

        # Get outputs
        for k, image_idx in enumerate(split_frame_numbers):
            fractional_time = image_idx
            if self.args.use_quads :
                fractional_time = image_idx / 4.0

            pose = split_pose[k]
            print(k, image_idx)

            self.render_kwargs_test['warp_offset'] = 0
            self.render_kwargs_test['eval_mode'] = True
            outputs = render(
                H=self.image_sizes['color'][1], W=self.image_sizes['color'][0],
                chunk=self.args.chunk,
                pose=pose, light_pose=pose,
                ray_gen_fn=self.generate_rays['color'],
                image_index=fractional_time,
                **self.render_kwargs_test
            )

            start_idx = 0
            start_iter = str(self.args.start_iter) if self.args.start_iter is not None else ''
            eval_dir = os.path.join(self.basedir, self.expname, 'eval_' + start_iter)

            os.makedirs(eval_dir, exist_ok=True)
            os.makedirs(eval_dir + '/eval', exist_ok=True)
            os.makedirs(eval_dir + '/eval_target', exist_ok=True)
            os.makedirs(eval_dir + '/eval_png', exist_ok=True)
            os.makedirs(eval_dir + '/eval_target_png', exist_ok=True)
            os.makedirs(eval_dir + '/eval_depth', exist_ok=True)
            os.makedirs(eval_dir + '/eval_target_depth', exist_ok=True)
            os.makedirs(eval_dir + '/eval_depth_png', exist_ok=True)
            os.makedirs(eval_dir + '/eval_target_depth_png', exist_ok=True)
            os.makedirs(eval_dir + '/eval_tof', exist_ok=True)
            os.makedirs(eval_dir + '/eval_target_tof', exist_ok=True)
            os.makedirs(eval_dir + '/eval_weights', exist_ok=True)
            os.makedirs(eval_dir + '/eval_z_vals', exist_ok=True)
            os.makedirs(eval_dir + '/eval_transmittance', exist_ok=True)
            os.makedirs(eval_dir + '/eval_tof_map_raw', exist_ok=True)
            os.makedirs(eval_dir + '/eval_rays', exist_ok=True)

            np.save(eval_dir + '/eval_rays/%04d_d.npy' % (k + start_idx), outputs['rays_d'])
            np.save(eval_dir + '/eval_rays/%04d_o.npy' % (k + start_idx), outputs['rays_o'])
            
            np.save(eval_dir + '/eval_weights/%04d.npy' % (k + start_idx), outputs['weights'])
            
            np.save(eval_dir + '/eval_depth/%04d.npy' % (k + start_idx), outputs['depth_map'])
            np.save(eval_dir + '/eval_z_vals/%04d.npy' % (k + start_idx), outputs['z_vals'])
            np.save(eval_dir + '/eval_target_depth/%04d.npy' % (k + start_idx), split_depths[k])
            np.save(eval_dir + '/eval_transmittance/%04d.npy' % (k + start_idx), outputs['transmittance'])
            np.save(eval_dir + '/eval_tof_map_raw/%04d.npy' % (k + start_idx), outputs['tof_map_raw'])

            disps = outputs['depth_map']
            disps = 1 - (disps - self.near) / (self.far - self.near)
            disps = cm.magma(disps)
            imageio.imwrite(
                eval_dir + '/eval_depth_png/%04d.png' % (k + start_idx), to8b(disps)
            )
            disps = split_depths[k]
            disps = 1 - (disps - self.near) / (self.far - self.near)
            disps = cm.magma(disps)
            imageio.imwrite(
                eval_dir + '/eval_target_depth_png/%04d.png' % (k + start_idx), to8b(disps)
            )

            np.save(eval_dir + '/eval/%04d.npy' % (k + start_idx), outputs['color_map'])
            np.save(eval_dir + '/eval_target/%04d.npy' % (k + start_idx), split_colors[k])
            np.save(eval_dir + '/eval_tof/%04d.npy' % (k + start_idx), outputs['tof_map'])
            np.save(eval_dir + '/eval_target_tof/%04d.npy' % (k + start_idx), split_tofs[k])
            
            imageio.imwrite(
                eval_dir + '/eval_png/%04d.png' % (k + start_idx), to8b(outputs['color_map'])
            )
            imageio.imwrite(
                eval_dir + '/eval_target_png/%04d.png' % (k + start_idx), to8b(split_colors[k])
            )

    def video_logging(self, i, mode='train', spiral_radius=0.05):
        print("video_logging mode:", mode)
        synthetic_dataset = self.args.gt_data_dir is not None and "synthetic" in self.args.gt_data_dir
        def rectify_poses(poses):
            for idx in range(poses.shape[0] // 4):
                for k in range(3):
                    poses[idx * 4 + (k + 1)] = poses[idx * 4]
            return poses
        
        def add_text_to_image(img, text):
                img_pil = Image.fromarray((img[..., :3] * 255).astype(np.uint8))
                I = ImageDraw.Draw(img_pil)
                I.text((0,0), text, fill=(255,0,0))
                img_with_text = img.copy()
                img_with_text[..., :3] = np.array(img_pil) / 255.0
                return img_with_text
        
        def add_text_to_video(video, text):
            video = [add_text_to_image(img, text) for img in video]
            return np.array(video)

        if self.args.dynamic:
            self.render_kwargs_test['warp_offset'] = 0
        render_poses_aux = None

        # Intended for spiral NVS on synthetic data
        use_spiral_extrinsics_file = ((self.args.render_extrinsics_file_spiral != "") and (mode == "manual_spiral"))

        if self.args.render_extrinsics_file != "":
            temp_pose = np.load(self.args.render_extrinsics_file)
            split_pose = np.tile(np.eye(4)[None], (temp_pose.shape[0], 1, 1))
            split_pose[:, :3, :] = temp_pose[:, :3, :4]

            split_pose = np.linalg.inv(split_pose)

            split_pose[:, :3, -1] *= self.args.render_extrinsics_scale
            split_pose, _ = recenter_poses(split_pose)

            if self.args.reverse_render_extrinsics:
                split_pose = split_pose[::-1]

            render_poses = split_pose
            render_light_poses = np.copy(split_pose)
        elif use_spiral_extrinsics_file : 
            render_extrinsics_filename = os.path.join(os.path.join(self.args.datadir, self.args.scan),self.args.render_extrinsics_file_spiral)
            temp_pose = np.load(render_extrinsics_filename)

            render_poses = temp_pose
            render_light_poses = temp_pose
        elif self.args.render_test:
            render_poses = np.array(
                self.color_poses[self.dataloader.i_test]
                )
            render_light_poses = np.copy(render_poses)
            render_poses_aux = np.array(
                self.tof_poses[self.dataloader.i_test]
                )

        elif 'spiral' in mode or 'freezeframe' in mode:
            render_poses, render_light_poses = \
                get_render_poses_spiral(
                    self.args.focus_distance,
                    self.dataset['bounds'], self.dataset['tof_intrinsics'],
                    self.tof_poses, self.args, 60, 2,
                    mode == 'manual_spiral', spiral_radius
                )
        elif mode == 'train':
            render_poses = np.array(
                [self.color_poses[idx] for idx in self.dataloader.i_train]
                )
            render_light_poses = np.copy(render_poses)
            render_poses_aux = np.array(
                [self.tof_poses[idx] for idx in self.dataloader.i_train]
            )
        else:
            raise NotImplementedError
            
        if self.args.dynamic:
            self.render_kwargs_test['outputs'] = [
                'tof_map', 'color_map',
                'tof_map_dynamic', 'color_map_dynamic',
                'disp_map', 'depth_map',
                'depth_map_dynamic', 'disp_map_dynamic',
                ] + (['forward_proj_map', 'backwards_proj_map', 'scene_flow_map',] if self.args.use_quads else [])

            self.render_kwargs_test['color_intrinsics'] = self.dataset['color_intrinsics'][0][:3, :3]
            self.render_kwargs_test['tof_intrinsics'] = self.dataset['tof_intrinsics'][0][:3, :3]

        cache_file = os.path.join(self.basedir, self.expname,'all_videos.pkl')
        cache_file_aux = os.path.join(self.basedir, self.expname,'all_videos_tof_pose.pkl')

        if self.args.clear_cache:
            if os.path.exists(cache_file):
                os.remove(cache_file)
            if os.path.exists(cache_file_aux):
                os.remove(cache_file_aux)
            print("Cache cleared.")

        if self.args.cache_outputs and os.path.exists(os.path.join(self.basedir, self.expname,'all_videos.pkl')):
            print("Loading cached outputs...")
            all_videos = pickle.load(open(cache_file, 'rb'))
        else:
            if not use_spiral_extrinsics_file :
                render_poses = rectify_poses(render_poses)

            all_videos = render_path(
                self.dataloader.i_train,
                dataset=self.dataset, all_poses=self.tof_poses, 
                render_poses=render_poses, light_poses=render_light_poses,
                ray_gen_fn_camera=self.generate_rays['color'], ray_gen_fn_light=self.generate_rays['color'],
                H=self.image_sizes['color'][1], W=self.image_sizes['color'][0],
                near=self.near, far=self.far, chunk=self.args.chunk // 16, render_kwargs=self.render_kwargs_test, args=self.args,
                render_freezeframe=(mode == 'freezeframe'),
                use_fractional_index=self.args.use_quads,
                all_output_names=self.render_kwargs_test['outputs'],
                )
            if self.args.cache_outputs:
                pickle.dump(all_videos, open(cache_file, 'wb'))

        if self.args.cache_outputs and os.path.exists(cache_file_aux):
            print("Loading cached outputs...")
            all_videos_tof_pose = pickle.load(open(cache_file_aux, 'rb'))
        else:
            if render_poses_aux is not None:
                render_poses_aux = rectify_poses(render_poses_aux)
                all_videos_tof_pose = render_path(
                    self.dataloader.i_train,
                    dataset=self.dataset, all_poses=self.tof_poses, 
                    render_poses=render_poses_aux, light_poses=render_light_poses,
                    ray_gen_fn_camera=self.generate_rays['tof'], ray_gen_fn_light=self.generate_rays['tof'],
                    H=self.image_sizes['tof'][1], W=self.image_sizes['tof'][0],
                    near=self.near, far=self.far, chunk= self.args.chunk // 16, render_kwargs=self.render_kwargs_test, args=self.args,
                    render_freezeframe=(mode == 'freezeframe'),
                    use_fractional_index=self.args.use_quads,
                    all_output_names=self.render_kwargs_test['outputs'],
                    )
                if self.args.cache_outputs:
                    pickle.dump(all_videos_tof_pose, open(cache_file_aux, 'wb'))
            else:
                all_videos_tof_pose = None

        # Write images
        image_base = mode
        depth_base = os.path.join(self.basedir, self.expname, image_base, 'depth_raw')
        rgb_base = os.path.join(self.basedir, self.expname, image_base, 'rgb_raw')
        if mode == 'manual_spiral':
            depth_base = os.path.join(depth_base,  "_spiral_", str(spiral_radius))
            rgb_base = os.path.join(rgb_base, "_spiral_", str(spiral_radius))

        os.makedirs(depth_base, exist_ok=True)
        os.makedirs(rgb_base, exist_ok=True)

        for j in range(all_videos['depth_map'].shape[0]):
            depth = all_videos['depth_map'][j]
            rgb = all_videos['color_map'][j]

            np.save(f'{depth_base}/{j:04d}.npy', depth)

            imageio.imwrite(
                f'{rgb_base}/{j:04d}.png',  cv2.cvtColor(to8b(rgb), cv2.COLOR_BGR2RGB)
            )

        # Write videos
        print('Done rendering videos, saving')
        
        if mode == 'train':
            # TODO: Choose only train data?
            train_indices = self.dataloader.i_train
            target_image = self.dataset['color_images'][train_indices]
            target_tof = self.dataset['tof_images'][train_indices]
            target_depth = self.dataset['depth_images'][train_indices]
            if self.args.use_quads:
                target_forward_flow = [self.dataset['forward_flow'][i] for i in train_indices]
                target_backward_flow = [self.dataset['backward_flow'][i] for i in train_indices]
                
                # Read blender target flows
                blender_forward_flow_dir = os.path.join(os.path.join(self.args.datadir, self.args.scan), "blender_forward_flow")
                blender_backward_flow_dir = os.path.join(os.path.join(self.args.datadir, self.args.scan), "blender_back_flow_blender")

                target_forward_flow_blender = []
                target_backward_flow_blender = []
                target_flow_shape = (2, self.args.color_image_height, self.args.color_image_width) # (2, H, W)
                
                def _read_flow(flow_filename):
                    try :
                        return np.load(flow_filename)
                    except OSError :
                        # print(f"NOT FOUND: {flow_filename}")
                        return np.zeros(target_flow_shape, dtype=np.float32)

                if  os.path.isdir(blender_backward_flow_dir) and os.path.isdir(blender_forward_flow_dir) :
                    for flow_idx in train_indices :
                        flow_idx = flow_idx - (flow_idx % 4)
                        forward_flow_filename = os.path.join(blender_forward_flow_dir, f"flow_{flow_idx:04d}.npy")
                        backward_flow_filename = os.path.join(blender_backward_flow_dir, f"flow_{flow_idx:04d}.npy")
                        
                        forward_flow = _read_flow(forward_flow_filename).transpose((1, 2, 0))
                        backward_flow = _read_flow(backward_flow_filename).transpose((1, 2, 0))

                        target_forward_flow_blender.append(forward_flow)
                        target_backward_flow_blender.append(backward_flow)
                
        prefix = ""
        if mode == 'manual_spiral':
            prefix = "manual_spiral_" + str(spiral_radius) + "_"
            if use_spiral_extrinsics_file :
                prefix = "manual_spiral_extrinsics_"
                
    
        moviebase_path = self.basedir, self.expname, f'{self.args.expname}_{image_base}_{i:08d}_'

        if synthetic_dataset:
            moviebase_path = [self.basedir, self.expname, f'{image_base}_{i:08d}_']
           
        moviebase = os.path.join(*moviebase_path)
        
        panel_video = []

        if all_videos_tof_pose is not None:
            tof_videos = all_videos_tof_pose
        else:
            tof_videos = all_videos

        if 'disp' in self.images_to_log:
            depths = tof_videos['depth_map']
            disps = 1 - (depths - self.near) / (self.far - self.near)
            disps_img = cm.magma(disps)

            disps_panel = disps_img

            quads = tof_videos['tof_map'][..., 3:]
            phase_offset = self.dataset['phase_offset'] if 'phase_offset' in self.dataset else 0
            
            if self.args.use_quads : # ToRF Baseline Support
                derived_depth = depth_from_quads(quads, self.dataset['depth_range'], phase_offset, perm=self.dataset['tof_permutation'])
                derived_depth = 1 - (derived_depth - self.near) / (self.far - self.near)
                derived_depth_img = cm.magma(derived_depth)
                derived_depth_panel = derived_depth_img

            if mode == 'train' or use_spiral_extrinsics_file:
                if use_spiral_extrinsics_file :
                    target_depth = [] 

                    for nvs_depth_idx in range(render_poses.shape[0]) :
                        nvs_depth_filename = os.path.join(self.args.datadir, self.args.scan, "synthetic_depth_nvs", f'{nvs_depth_idx:04d}.npy')
                        target_depth.append(np.load(nvs_depth_filename))

                    target_depth = np.array(target_depth)

                    assert target_depth.shape == (render_poses.shape[0], self.args.tof_image_height, self.args.tof_image_width) 

                disps_gt = 1 - (target_depth - self.near) / (self.far - self.near)

                disps_gt_img = cm.magma(disps_gt)

                disps_panel = np.concatenate([disps_gt_img, disps_img], axis=1)

                depth_diff = np.clip(np.abs(target_depth - depths) / (target_depth + 1e-3), 0, 1)
                depth_diff_img = cm.viridis(depth_diff)
                depth_diff_panel = add_text_to_video(np.concatenate([depth_diff_img, depth_diff_img], axis=1)[..., :3], "Depth diff")
                if self.args.use_quads : # ToRF Baseline Support
                    derived_depth_panel = np.concatenate([disps_gt_img, derived_depth_img], axis=1)
                imageio.mimwrite(
                    moviebase + prefix + 'depth_diff.mp4', to8b(depth_diff_img),
                    fps=25, quality=8
                )

                panel_video.append(depth_diff_panel)

                depth_diff_masks_video = []
                for threshold in [0.05, 0.1, 0.2, 0.5]:
                    depth_diff_threshold_idx = depth_diff > threshold
                    depth_diff_threshold_img = np.zeros_like(depth_diff_img)
                    depth_diff_threshold_img[depth_diff_threshold_idx] = 1
                    depth_diff_threshold_img = add_text_to_video(depth_diff_threshold_img, f"Depth error > {100 * threshold:.2f}%")
                    depth_diff_masks_video.append(depth_diff_threshold_img)
                depth_diff_masks_video = np.concatenate(depth_diff_masks_video, axis=2)[..., :3]

                imageio.mimwrite(
                    moviebase + prefix + 'depth_diff_masks.mp4', to8b(depth_diff_masks_video),
                    fps=25, quality=8
                )

                imageio.mimwrite(
                    moviebase + prefix + 'disp_GT.mp4', to8b(disps_gt_img),
                    fps=25, quality=8
                )

            panel_video.append(add_text_to_video(disps_panel[..., :3], "Depth"))
            if self.args.use_quads : # ToRF Baseline Support
                panel_video.append(add_text_to_video(derived_depth_panel[..., :3], "Derived depth"))

            imageio.mimwrite(
                moviebase + prefix + 'disp.mp4', to8b(disps_img),
                fps=25, quality=8
            )
            
            if self.args.use_quads : # ToRF Baseline Support
                imageio.mimwrite(
                    moviebase + prefix + 'derived_depth.mp4', to8b(derived_depth_img),
                    fps=25, quality=8
                )
            
            if self.args.dynamic:
                disps = tof_videos['depth_map_dynamic']
                disps = 1 - (disps - self.near) / (self.far - self.near)
                disps = cm.magma(disps)

                if len(disps):
                    imageio.mimwrite(
                        moviebase + prefix + 'disp_dynamic.mp4', to8b(disps),
                        fps=25, quality=8
                    )
        
        if 'tof_cos' in self.images_to_log:
            tofs = tof_videos['tof_map']
            tofs_cos = normalize_im(np.abs(tofs[..., 0]))
            tofs_cos = to8b(tofs_cos)
            imageio.mimwrite(
                moviebase + prefix + 'tof_cos.mp4', tofs_cos,
                fps=25, quality=8
            )

        if 'tof_sin' in self.images_to_log:
            tofs = tof_videos['tof_map']
            tofs_sin = normalize_im(np.abs(tofs[..., 1]))

            imageio.mimwrite(
                moviebase + prefix + 'tof_sin.mp4', to8b(tofs_sin),
                fps=25, quality=8
            )

        if ('tof_amp' in self.images_to_log) and (not use_spiral_extrinsics_file):
            tofs = tof_videos['tof_map']
            tofs_amp = normalize_im(np.abs(tofs[..., 2]))

            tofs_amp_panel = tofs_amp
            if mode == 'train':
                tofs_amp = normalize_im_gt(np.abs(tofs[..., 2]), np.abs(target_tof[..., 2]))
                tofs_amp_gt = normalize_im(np.abs(target_tof[..., 2]))
                tofs_amp_panel = np.concatenate([tofs_amp_gt, tofs_amp], axis=1)
            
            # Extend to 3 channels
            tofs_amp_panel = np.tile(tofs_amp_panel[..., None], (1, 1, 3))

            panel_video.append(add_text_to_video(tofs_amp_panel, "Amplitude"))

            imageio.mimwrite(
                moviebase + prefix + 'tof_amp.mp4', to8b(tofs_amp),
                fps=25, quality=8
            )

            if self.args.dynamic:
                tofs = tof_videos['tof_map_dynamic']
                if len(tofs):
                    tofs_amp = normalize_im(np.abs(tofs[..., 1]))

                    imageio.mimwrite(
                        moviebase + prefix + 'tof_amp_dynamic.mp4', to8b(tofs_amp),
                        fps=25, quality=8
                    )

        if 'color' in self.images_to_log:
            colors = all_videos['color_map']

            colors_processed = to8b(colors)
            
            # flip channels for synthetic data
            if synthetic_dataset:
                colors_processed_temp = []
                for j in range(colors_processed.shape[0]) :
                    colors_processed_temp.append(cv2.cvtColor(colors[j], cv2.COLOR_BGR2RGB))
                colors_processed = np.array(colors_processed)

            imageio.mimwrite(
                moviebase + prefix + 'color.mp4', colors_processed,
                fps=25, quality=8
            )

            colors_panel = colors
            if mode == 'train' or use_spiral_extrinsics_file:
                if self.args.use_quads :
                    if use_spiral_extrinsics_file :
                        colors_gt = [] 

                        for nvs_color_idx in range(render_poses.shape[0]) :
                            nvs_color_filename = os.path.join(self.args.datadir, self.args.scan, "color_nvs", f'{nvs_color_idx:04d}.npy')
                            colors_gt.append(np.load(nvs_color_filename))

                        colors_gt = np.array(colors_gt) * self.dataset["color_norm_factor"]

                        assert colors_gt.shape == (render_poses.shape[0], self.args.color_image_height, self.args.color_image_width, 3)  
                    else :
                        colors_gt = tof_fill_blanks(target_image[..., :3], 0)
                        


                else :
                    colors_gt = target_image[..., :3] # ToRF baseline support

                colors_panel = np.concatenate([colors_gt, colors], axis=1)
            

            colors_processed_panel = add_text_to_video(colors_panel, "Color")

            # [Hack] Flipping color channels for synthetic vidoes
            if synthetic_dataset:
                colors_processed_temp_panel = []
                for j in range(colors_panel.shape[0]) :
                    colors_processed_temp_panel.append(
                        cv2.cvtColor(add_text_to_image(colors_panel[j], "Color"), cv2.COLOR_BGR2RGB)
                    )
                colors_processed_panel = np.array(colors_processed_temp_panel)            
                
            panel_video.append(colors_processed_panel)

            if self.args.dynamic:
                colors = all_videos['color_map_dynamic']
                if len(colors):
                    imageio.mimwrite(
                        moviebase + prefix + 'color_dynamic.mp4', to8b(colors),
                        fps=25, quality=8
                    )
        tof_labels = ['cos', '-cos', 'sin', '-sin']
        tof_labels = [tof_labels[self.dataset['tof_inverse_permutation'][idx]] for idx in range(4)]

        def tof_to_img(tof):
            cmap = plt.get_cmap('seismic')
            norm = plt.Normalize(-0.5, 0.5)

            if abs(self.dc_offset - 0.0) > 1e-5 :
                # [Hack] Better viz on synthetic quads because range is shifted
                norm = plt.Normalize(0.0, 1.0)
            tof_rgb = cmap(norm(tof))[:, :, :3]
            return tof_rgb

        for quad in range(4):
            if (f'tofQuad_{quad}' in self.images_to_log) and (not use_spiral_extrinsics_file):
                tofs = tof_videos['tof_map']
                tof_quad = tofs[..., 3 + quad]
                
                if mode == 'train':
                    tof_quad_target = self.dataset[f'tofQuad{quad}_images'][train_indices, ..., 0]
                    tof_quad_target = tof_fill_blanks(tof_quad_target, quad)
                    tof_quad_target = np.stack([tof_to_img(tof_quad_target[i]) for i in range(tof_quad_target.shape[0])], axis=0)
                    tof_quad = np.stack([tof_to_img(tof_quad[i]) for i in range(tof_quad.shape[0])], axis=0)

                    tof_quad_panel = np.concatenate([tof_quad_target, tof_quad], axis=1)
                else:
                    tof_quad_panel = np.stack([tof_to_img(tof_quad[i]) for i in range(tof_quad.shape[0])], axis=0)
                
                # Extend to 3 channels
                panel_video.append(add_text_to_video(tof_quad_panel, f"{tof_labels[quad]}"))

                imageio.mimwrite(
                    moviebase + prefix + f'tof_{quad}.mp4', to8b(tof_quad), 
                    fps=25, quality=8
                )

        if ('scene_flow' in self.images_to_log) and (not use_spiral_extrinsics_file):
            def make_flow_error_viz_videos(flows, flows_gt, flow_err_magnitude_threshold=10):
                # Inputs: flows: [N, W, H, 2] -- predicted flows
                #         flows_gt: [N, W, H, 2] -- ground truth flows
                # Outputs: dict of videos

                error_flow = flows_gt - flows
                error_flow_magnitude = np.clip(np.linalg.norm(error_flow, axis=-1), 0, flow_err_magnitude_threshold)
                error_flow_videos = draw_flow_video(error_flow, flows_gt)
                error_flow_magnitude_videos = cm.viridis(error_flow_magnitude / flow_err_magnitude_threshold)[..., :3]
                error_flow_magnitude_thresholded_videos = []
                for threshold in [1.0, 2.0, 5.0, 10.0]:
                    thresholded_video = np.where(error_flow_magnitude > threshold, 1.0, 0.0)
                    # Convert to 3-channel video
                    thresholded_video = np.stack([thresholded_video, thresholded_video, thresholded_video], axis=-1)
                    error_flow_magnitude_thresholded_videos.append((thresholded_video, threshold))

                return {
                    "error_flow": error_flow_videos,
                    "error_flow_magnitude": error_flow_magnitude_videos,
                    "error_flow_magnitude_thresholded": error_flow_magnitude_thresholded_videos
                }
            
            def make_flow_viz_panel(flows, flows_gt, flow_err_magnitude_threshold=10, flow_source= "RAFT Flow"):
                flow_panel_video = []
                flow_viz = draw_flow_video(flows, flows_gt)
                flow_gt_viz = draw_flow_video(flows_gt)
                flow_err_videos = make_flow_error_viz_videos(flows, flows_gt, flow_err_magnitude_threshold)

                flow_panel = np.concatenate([flow_gt_viz, flow_viz], axis=1)
                flow_panel_video.append(add_text_to_video(flow_panel, flow_source))

                flow_err_panel = np.concatenate([flow_gt_viz, flow_err_videos['error_flow']], axis=1)
                flow_panel_video.append(add_text_to_video(flow_err_panel, "Flow error"))

                flow_err_magnitude_panel = np.concatenate([flow_gt_viz, flow_err_videos['error_flow_magnitude']], axis=1)
                flow_panel_video.append(add_text_to_video(flow_err_magnitude_panel, "Flow error magnitude"))

                for thresholded_video, threshold in flow_err_videos['error_flow_magnitude_thresholded']:
                    thresholded_video_panel = np.concatenate([flow_gt_viz, thresholded_video], axis=1)
                    flow_panel_video.append(add_text_to_video(thresholded_video_panel, f"Flow error > {threshold:.2f} px"))

                return np.concatenate(flow_panel_video, axis=2)

            # TODO(mokunev): figure out what's wrong with tof pos sf render
            #width, height = self.image_sizes['tof']
            width, height = self.image_sizes['color']
            source_coordinates = np.array(np.meshgrid(
                                    np.linspace(0., width - 1., width),
                                    np.linspace(0., height - 1., height),
                                    )).transpose((1, 2, 0))

            # TODO(mokunev): figure out what's wrong with tof pos sf render
            #target_coordinates_forward = all_videos_tof_pose['forward_proj_map']
            #target_coordinates_backwards = all_videos_tof_pose['backwards_proj_map']

            target_coordinates_forward = all_videos['forward_proj_map']
            target_coordinates_backwards = all_videos['backwards_proj_map']

            forward_flow = target_coordinates_forward - source_coordinates
            backwards_flow = target_coordinates_backwards - source_coordinates

            forward_flow_viz = draw_flow_video(forward_flow)
            backwards_flow_viz = draw_flow_video(backwards_flow)

            forward_flow_panel = forward_flow_viz
            backwards_flow_panel = backwards_flow_viz
            if mode == 'train':
                forward_flow_viz = draw_flow_video(forward_flow, target_forward_flow)
                backwards_flow_viz = draw_flow_video(backwards_flow, target_backward_flow)

                forward_flow_gt_viz = draw_flow_video(target_forward_flow)
                backwards_flow_gt_viz = draw_flow_video(target_backward_flow)

                forward_flow_panel = np.concatenate([forward_flow_gt_viz, forward_flow_viz], axis=1)
                backwards_flow_panel = np.concatenate([backwards_flow_gt_viz, backwards_flow_viz], axis=1)

                if target_backward_flow_blender and target_forward_flow_blender :
                    forward_flow_viz_blender = draw_flow_video(forward_flow, target_forward_flow_blender)
                    backwards_flow_viz_blender = draw_flow_video(backwards_flow, target_backward_flow_blender)

                    forward_flow_gt_viz_blender = draw_flow_video(target_forward_flow_blender)
                    backwards_flow_gt_viz_blender = draw_flow_video(target_backward_flow_blender)

                    forward_flow_panel_blender = np.concatenate([forward_flow_gt_viz_blender, forward_flow_viz_blender], axis=1)
                    backwards_flow_panel_blender = np.concatenate([backwards_flow_gt_viz_blender, backwards_flow_viz_blender], axis=1)

                    panel_video.append(add_text_to_video(forward_flow_panel_blender, "Blender Forward Flow"))
                    panel_video.append(add_text_to_video(backwards_flow_panel_blender, "Blender Backwards flow"))

                forward_flow_err_viz_panel = make_flow_viz_panel(forward_flow, target_forward_flow)
                backwards_flow_err_viz_panel = make_flow_viz_panel(backwards_flow, target_backward_flow)

                imageio.mimwrite(
                    moviebase + prefix + 'forward_flow_err_panel.mp4', to8b(forward_flow_err_viz_panel),
                    fps=25, quality=8
                )

                imageio.mimwrite(
                    moviebase + prefix + 'backwards_flow_err_panel.mp4', to8b(backwards_flow_err_viz_panel),
                    fps=25, quality=8
                )

                if target_backward_flow_blender and target_forward_flow_blender : 
                    # Generate error BLENDER GT vs ToRF++
                    forward_flow_err_viz_panel = make_flow_viz_panel(forward_flow, target_forward_flow_blender, flow_source="Blender Flow")
                    backwards_flow_err_viz_panel = make_flow_viz_panel(backwards_flow, target_backward_flow_blender, flow_source="Blender Flow")

                    imageio.mimwrite(
                        moviebase + prefix + 'blender_forward_flow_err_panel.mp4', to8b(forward_flow_err_viz_panel),
                        fps=25, quality=8
                    )

                    imageio.mimwrite(
                        moviebase + prefix + 'blender_backwards_flow_err_panel.mp4', to8b(backwards_flow_err_viz_panel),
                        fps=25, quality=8
                    )

                    # Generate error BLENDER GT vs RAFT
                    forward_flow_err_viz_panel = make_flow_viz_panel(np.array(target_forward_flow), target_forward_flow_blender, flow_source="Blender vs RAFT")
                    backwards_flow_err_viz_panel = make_flow_viz_panel(np.array(target_backward_flow), target_backward_flow_blender, flow_source="Blender vs RAFT")

                    imageio.mimwrite(
                        moviebase + prefix + 'blender_raft_forward_flow_err_panel.mp4', to8b(forward_flow_err_viz_panel),
                        fps=25, quality=8
                    )

                    imageio.mimwrite(
                        moviebase + prefix + 'blender_raft_backwards_flow_err_panel.mp4', to8b(backwards_flow_err_viz_panel),
                        fps=25, quality=8
                    )


            # General panel
            panel_video.append(add_text_to_video(forward_flow_panel, "Forward flow"))
            panel_video.append(add_text_to_video(backwards_flow_panel, "Backwards flow"))

            imageio.mimwrite(
                moviebase + prefix + 'forward_flow.mp4', to8b(forward_flow_viz),
                fps=25, quality=8
            )

            imageio.mimwrite(
                moviebase + prefix + 'backwards_flow.mp4', to8b(backwards_flow_viz),
                fps=25, quality=8
            )

        if len(panel_video):
            len_panel_video = len(panel_video)
            reshape_video = len_panel_video > 6
            if reshape_video :
                if len_panel_video % 2 == 1 : # make len(panel_video) even
                    panel_video.append(panel_video[-1])

            panel_video = np.concatenate(panel_video, axis=2)

            if reshape_video :
                video_width = panel_video.shape[2]
                top_half = panel_video[:, :, : video_width // 2, :]
                bottom_half = panel_video[:, :, video_width // 2 :, :]
                panel_video = np.concatenate([top_half, bottom_half], axis=1)

            imageio.mimwrite(
                moviebase + prefix + 'panel.mp4', to8b(panel_video),
                fps=25, quality=8
            )
    
    def metric_logging(self, i):
        # Computes Depth MSE, Masked DEPTH MSE, PSNR for integer frames

        # Copy union_dynamic_ROI_mask and synthetic depth into each folder
        print(f'Computing metrics; Experiment name {self.args.expname}', flush=True)
        depth_mse_total = 0.0 # full image
        dynamic_depth_mse_total = 0 # dynamic region
        psnr_total = 0.0

        derived_depth_mse_total = 0.0 # full image
        derived_dynamic_depth_mse_total = 0 # dynamic region

        train_indices = self.dataloader.i_train[::4]
        train_indices = train_indices[1:-1]

        print(train_indices, flush=True)

        ctof_depth_directory = os.path.join(
            "/ifs/CS/replicated/data/jhtlab/mmapeke/tof_nerf_split/data_final/legacy_torf/",
            self.args.scan, "depth")
        
        ctof_depth_mse_total = 0.0 # full image
        ctof_dynamic_depth_mse_total = 0 # dynamic region
        
        warp_depth_directory = os.path.join(
            "/ifs/CS/replicated/data/jhtlab/mmapeke/tof_nerf_split/data_final/2D_warped_softsplatinterp_withexistingflow/",
            self.args.scan, "depth")

        warp_depth_mse_total = 0.0 # full image
        warp_dynamic_depth_mse_total = 0 # dynamic region

        # Collect for motion maps
        target_forward_flow = [self.dataset['forward_flow'][i] for i in train_indices]
        target_backward_flow = [self.dataset['backward_flow'][i] for i in train_indices]
        print("Length of target flows", len(target_forward_flow), len(target_backward_flow))

        num_frames = len(train_indices)
        
        disps_img = [] 
        disps_derived_img = [] 
        disps_gt_img = []
        disps_warp_img = []
        disps_ctof_img = []

        for img_i in train_indices :
            dynamic_mask = cv2.imread(os.path.join(
                self.args.datadir,
                f'{self.args.scan}/union_dynamic_ROI_mask/{img_i:04d}.png' 
                ))

            dynamic_mask = dynamic_mask[..., 0]
            dynamic_mask = dynamic_mask < 0.0001 # True = static
            
            target_image = self.dataset['color_images'][img_i]

            # Load depth from dataset with GT
            target_depth = np.load(
                os.path.join(
                self.args.datadir,
                f'{self.args.scan}/{self.args.gt_data_dir}/{img_i:04d}.npy' 
                )
            )
            target_depth = target_depth.squeeze()
            print("target (gt) depth directory:", 
                  os.path.join(
                    self.args.datadir,
                    f'{self.args.scan}/{self.args.gt_data_dir}/{img_i:04d}.npy' 
                    ) )

            
            pose = self.tof_poses[img_i]
            light_pose = self.tof_light_poses[img_i]

            if self.args.dynamic:
                self.render_kwargs_test['warp_offset'] = 0
                self.render_kwargs_test['outputs'] = [
                    'tof_map', 'color_map',
                    'depth_map', 'disp_map',
                    'tof_map_dynamic', 'color_map_dynamic',
                    'depth_map_dynamic', 'disp_map_dynamic',
                ] + ['scene_flow_map', 'forward_proj_map', 'backwards_proj_map']

                # Storing information needed to project SF to OF
                self.render_kwargs_test['color_intrinsics'] = self.dataset['color_intrinsics'][0][:3, :3]
                self.render_kwargs_test['backwards_pose'] = self.color_poses[max(img_i - 4, 0)]
                self.render_kwargs_test['forward_pose'] = self.color_poses[min(img_i + 4, len(self.color_poses) - 1)]

            fractional_img_i = img_i
            if self.args.dataset_type == 'quads': 
                fractional_img_i = img_i / 4.0 

            outputs = render(
                H=self.image_sizes['tof'][1], W=self.image_sizes['tof'][0],
                chunk=int(1000 * self.args.chunk),
                pose=pose, light_pose=light_pose,
                ray_gen_fn=self.generate_rays['tof'],
                image_index=fractional_img_i,
                **self.render_kwargs_test
            )

            # Save out the validation image for Tensorboard-free monitoring
            testimgdir = os.path.join(self.basedir, self.expname, 'metric_logging')
            predicted_color_imgdir = os.path.join(testimgdir, "predicted_color")
            predicted_depth_imgdir = os.path.join(testimgdir, 'predicted_depth')
            predicted_forward_motion_imgdir = os.path.join(testimgdir, "forward_motion_map")
            predicted_backward_motion_imgdir = os.path.join(testimgdir, "backward_motion_map")
            disp_videodir = os.path.join(testimgdir, "disp_videos")

        
            os.makedirs(testimgdir, exist_ok=True)  
            os.makedirs(predicted_depth_imgdir, exist_ok=True)
            os.makedirs(predicted_color_imgdir, exist_ok=True)
            os.makedirs(predicted_forward_motion_imgdir, exist_ok=True)
            os.makedirs(predicted_backward_motion_imgdir, exist_ok=True)
            os.makedirs(disp_videodir, exist_ok=True)


            # Based off Example from image_logging

            # Scene Flow Projected to Optical Flow, Use OF color wheel
            width, height = self.image_sizes['color']

            source_coordinates = np.array(np.meshgrid(
                                    np.linspace(0., width - 1., width),
                                    np.linspace(0., height - 1., height),
                                    )).transpose((1, 2, 0))

            target_coordinates_forward = outputs['forward_proj_map']
            target_coordinates_backwards = outputs['backwards_proj_map']

            forward_flow = target_coordinates_forward - source_coordinates
            backwards_flow = target_coordinates_backwards - source_coordinates

            forward_flow_viz = flow_to_image(forward_flow, target_forward_flow)
            backwards_flow_viz = flow_to_image(backwards_flow, target_backward_flow)

            imageio.imwrite(
                os.path.join(predicted_forward_motion_imgdir, '{:04d}_motion_forwards.png'.format(img_i)), forward_flow_viz
            )
            imageio.imwrite(
                os.path.join(predicted_backward_motion_imgdir, '{:04d}_motion_backwards.png'.format(img_i)), backwards_flow_viz
            )

            # Depth (depth mse, dynamic depth mse)
            disp = 1 - (outputs['depth_map'] - self.near) / (self.far - self.near)
            disp = cm.magma(disp)
            disps_img.extend([disp] * 4)

            disp_gt = 1 - (target_depth - self.near) / (self.far - self.near)
            disp_gt = cm.magma(disp_gt)
            disps_gt_img.extend([disp_gt] * 4)

            mse = np.mean((outputs['depth_map'] - target_depth)**2) # No crop
            dynamic_mse = np.mean( np.ma.masked_where( dynamic_mask  , ((outputs['depth_map'] - target_depth)**2))) # dynamic region
        
            depth_mse_total = depth_mse_total + mse
            dynamic_depth_mse_total = dynamic_depth_mse_total + dynamic_mse

            # Saving depth imgs
            def normalize(a: np.array, a_min: float, a_max: float, out_min: float, out_max: float, force_normalization=False):
                if a_min is None:
                    a_min = np.min(a)
                if a_max is None:
                    a_max = np.max(a)
                a_unit = (a - a_min) / (a_max - a_min)
                a_unit = np.clip(a_unit, 0, 1)

                return a_unit * (out_max - out_min) + out_min

            # Current ToRF Variant
            cv2.imwrite(os.path.join(predicted_depth_imgdir, f"{img_i:04d}_error_depth.png"), img=normalize(np.abs(np.array(outputs['depth_map']) - target_depth), 
                                                                                                            None, None, 0, 255))
            imageio.imwrite(
                os.path.join(predicted_depth_imgdir, '{:04d}_disp.png'.format(img_i)), to8b(disp)
            )
            imageio.imwrite(
                os.path.join(predicted_depth_imgdir, '{:04d}_disp_gt.png'.format(img_i)), to8b(disp_gt)
            )

            # Color (PSNR)
            imageio.imwrite(
                os.path.join(predicted_color_imgdir, '{:04d}_color.png'.format(img_i)), cv2.cvtColor(to8b(outputs['color_map']), cv2.COLOR_BGR2RGB)
            )
            imageio.imwrite(
                os.path.join(predicted_color_imgdir, '{:04d}_color_gt.png'.format(img_i)), cv2.cvtColor(to8b(target_image), cv2.COLOR_BGR2RGB)
            )
            
            _img_loss = np.array(((self.dataset['color_images'][img_i][..., :3] - outputs['color_map'])**2)).mean()
            _img_loss = np.float32(_img_loss)

            psnr = mse2psnr(_img_loss).numpy()
            psnr_total = psnr_total + psnr

            print(img_i, "Depth MSE", mse, "Dynamic MSE", dynamic_mse, "PSNR", psnr)


            # CTOF
            print(img_i, img_i // 4)
            depth_ctof = np.load(
                os.path.join(ctof_depth_directory, f"{(img_i // 4):04d}.npy")
            )
            
            disp_ctof_viz = 1 - (depth_ctof - self.near) / (self.far - self.near) 
            disp_ctof_viz = cm.magma(disp_ctof_viz)
            disps_ctof_img.extend([disp_ctof_viz] * 4)

            ctof_mse = np.mean((depth_ctof- target_depth)**2) # No crop
            ctof_dynamic_mse = np.mean( np.ma.masked_where( dynamic_mask  , ((depth_ctof - target_depth)**2))) # dynamic region
            
            ctof_depth_mse_total += ctof_mse
            ctof_dynamic_depth_mse_total += ctof_dynamic_mse

            imageio.imwrite(
                os.path.join(predicted_depth_imgdir, '{:04d}_disp_ctof.png'.format(img_i)), to8b(disp_ctof_viz)
            )

            # 2d Warped
            depth_warp = np.load(
                os.path.join(warp_depth_directory, f"{img_i // 4:04d}.npy")
            )
            depth_warp = depth_warp.squeeze()

            disp_warp_viz = 1 - (depth_warp - self.near) / (self.far - self.near) 
            disp_warp_viz = cm.magma(disp_warp_viz)
            disps_warp_img.extend([disp_warp_viz] * 4)

            warp_mse = np.mean((depth_warp - target_depth)**2) # No crop
            warp_dynamic_mse = np.mean( np.ma.masked_where( dynamic_mask  , ((depth_warp - target_depth)**2))) # dynamic region

            warp_depth_mse_total += warp_mse 
            warp_dynamic_depth_mse_total += warp_dynamic_mse

            imageio.imwrite(
                os.path.join(predicted_depth_imgdir, '{:04d}_disp_2d_warp.png'.format(img_i)), to8b(disp_warp_viz)
            )

            # Derived depth 
            quads =  outputs['tof_map'][..., 3:]
            phase_offset = self.dataset['phase_offset'] if 'phase_offset' in self.dataset else 0
            
            if self.args.use_quads : # ToRF Baseline Support
                derived_depth = depth_from_quads(np.array(quads), self.dataset['depth_range'], phase_offset, perm=self.dataset['tof_permutation'])
                derived_depth_img = 1 - (derived_depth - self.near) / (self.far - self.near)
                derived_depth_img = cm.magma(derived_depth_img)
                disps_derived_img.extend([derived_depth_img] * 4)

            derived_mse = np.mean((derived_depth - target_depth)**2) # No crop
            derived_dynamic_mse = np.mean( np.ma.masked_where( dynamic_mask  , ((derived_depth - target_depth)**2))) # dynamic region

            derived_depth_mse_total += derived_mse
            derived_dynamic_depth_mse_total += derived_dynamic_mse

            imageio.imwrite(
                os.path.join(predicted_depth_imgdir, '{:04d}_derived_2d_warp.png'.format(img_i)), to8b(derived_depth_img)
            )

        disps_img = np.array(disps_img)
        disps_gt_img = np.array(disps_gt_img)
        disps_ctof_img = np.array(disps_ctof_img)
        disps_warp_img = np.array(disps_warp_img)
        disps_derived_img = np.array(disps_derived_img)

        print(disps_img.shape, disps_gt_img.shape, disps_ctof_img.shape, disps_warp_img.shape, disps_derived_img.shape)

        imageio.mimwrite(
                os.path.join(disp_videodir, 'predicted_disp.mp4'), to8b(disps_img),
                fps=25, quality=8
            )
        
        gt_filename_suffix = ""
        if 'blender' in self.args.gt_data_dir :
            gt_filename_suffix = "_blender"

        imageio.mimwrite(
                os.path.join(disp_videodir, f'gt_disp{gt_filename_suffix}.mp4'), to8b(disps_gt_img),
                fps=25, quality=8
            )
        
        imageio.mimwrite(
                os.path.join(disp_videodir, 'ctof_disp.mp4'), to8b(disps_ctof_img),
                fps=25, quality=8
            )

        imageio.mimwrite(
                os.path.join(disp_videodir, '2d_warp_disp.mp4'), to8b(disps_warp_img),
                fps=25, quality=8
            )
        
        imageio.mimwrite(
                os.path.join(disp_videodir, 'predicted_derived_disp.mp4'), to8b(disps_derived_img),
                fps=25, quality=8
            )
      
        def metrics_format_helper(value) :
            result = value * 100
            rounded_result = round(result, 3)
            return rounded_result


        print()
        print("avg depth mse", metrics_format_helper( depth_mse_total / float(num_frames)))
        print("avg dynamic depth mse", metrics_format_helper( dynamic_depth_mse_total / float(num_frames)))
        print("avg psnr", psnr_total / float(num_frames))
        print()
        print("(derived) avg depth mse", metrics_format_helper( derived_depth_mse_total / float(num_frames)))
        print("(derived) avg dynamic depth mse", metrics_format_helper( derived_dynamic_depth_mse_total / float(num_frames)))
        print()
        print("CTOF avg depth mse", metrics_format_helper( ctof_depth_mse_total / float(num_frames)))
        print("CTOF avg dynamic depth mse", metrics_format_helper( ctof_dynamic_depth_mse_total / float(num_frames)))
        print()
        print("2d warp avg depth mse", metrics_format_helper( warp_depth_mse_total / float(num_frames)))
        print("2d warp avg dynamic depth mse", metrics_format_helper( warp_dynamic_depth_mse_total / float(num_frames)))

    def image_logging(self, i):
        print("Starting image logging...")
        if (i % (2 * self.args.i_img) == 0):
            i_choice = self.dataloader.i_train
            suffix = "_train"
        else:
            i_choice = self.dataloader.i_test
            suffix = ""

        # Log a rendered validation view to Tensorboard
        img_i = np.random.choice(i_choice)

        # Visualizing one frame for debugging, should be less than 4 to make 4-frame sequences work
        if self.args.use_quads :
            img_i = 2 * 4
        else :
            img_i = 2

        target_image = self.dataset['color_images'][img_i]
        target_tof = self.dataset['tof_images'][img_i]
        target_depth = self.dataset['depth_images'][img_i]

        if 'scene_flow' in self.images_to_log :
            target_forward_flow = self.dataset['forward_flow'][img_i]
            target_backward_flow = self.dataset['backward_flow'][img_i]

        tof_pose = self.tof_poses[img_i]
        color_pose = self.color_poses[img_i]
        light_pose = self.tof_light_poses[img_i]

        if self.args.dynamic:
            self.render_kwargs_test['warp_offset'] = 0
            self.render_kwargs_test['outputs'] = [
                'tof_map', 'color_map',
                'depth_map', 'disp_map',
                'tof_map_dynamic', 'color_map_dynamic', 'color_map_static',
                'depth_map_dynamic', 'disp_map_dynamic', 'disp_map_static',
            ] + (['scene_flow_map', 'depth_map', 'forward_proj_map', 'backwards_proj_map'] if self.args.scene_flow else []) + (['disocclusion_map'] if self.args.disocclusion else [])

            # Storing information needed to project SF to OF
            self.render_kwargs_test['color_intrinsics'] = self.dataset['color_intrinsics'][0][:3, :3]
            self.render_kwargs_test['tof_intrinsics'] = self.dataset['tof_intrinsics'][0][:3, :3]
            self.render_kwargs_test['backwards_pose'] = self.color_poses[img_i - 4]
            self.render_kwargs_test['forward_pose'] = self.color_poses[img_i + 4]

        fractional_img_i = img_i
        if self.args.dataset_type == 'quads': 
            fractional_img_i = img_i / 4.0 

        outputs_tof_pose = render(
            H=self.image_sizes['tof'][1], W=self.image_sizes['tof'][0],
            chunk=int(self.args.chunk / 32),
            pose=tof_pose, light_pose=light_pose,
            ray_gen_fn=self.generate_rays['tof'],
            image_index=fractional_img_i,
            **self.render_kwargs_test,
        )

        outputs_color_pose = render(
            H=self.image_sizes['color'][1], W=self.image_sizes['color'][0],
            chunk=int(self.args.chunk / 32),
            pose=color_pose, light_pose=light_pose,
            ray_gen_fn=self.generate_rays['color'],
            image_index=fractional_img_i,
            **self.render_kwargs_test,
        )
        # Save out the validation image for Tensorboard-free monitoring
        testimgdir = os.path.join(self.basedir, self.expname, 'tboard_val_imgs')
        sceneflowimgdir = os.path.join(testimgdir, 'scene_flow')
        partial_sceneflowimgdir = os.path.join(testimgdir, 'partial_scene_flow')
        occlusionimgdir = os.path.join(testimgdir, 'disocclusion')
        weights_dir = os.path.join(testimgdir, 'weights')

        os.makedirs(testimgdir, exist_ok=True)
        os.makedirs(sceneflowimgdir, exist_ok=True)
        os.makedirs(partial_sceneflowimgdir, exist_ok=True)
        os.makedirs(occlusionimgdir, exist_ok=True)
        os.makedirs(weights_dir, exist_ok=True)

        if 'weights' in outputs_tof_pose:
            np.save(os.path.join(weights_dir, '{:08d}_weights.npy'.format(i)), outputs_tof_pose['weights'])

        # Write images
        if 'disp' in self.images_to_log:
            disp = 1 - (outputs_tof_pose['depth_map'] - self.near) / (self.far - self.near)
            disp = cm.magma(disp)

            disp_gt = 1 - (target_depth - self.near) / (self.far - self.near)
            disp_gt = cm.magma(disp_gt)

            disp_diff = np.abs(disp - disp_gt) / disp_gt

            # Adding histograms
            fig = plt.hist(np.array(np.clip(outputs_tof_pose['depth_map'], 0, 16)).flatten(), bins=500)
            plt.savefig(os.path.join(testimgdir, '{:08d}_disp_hist.png'.format(i)))
            plt.clf()

            fig = plt.hist(np.array(target_depth).flatten(), bins=500)
            plt.savefig(os.path.join(testimgdir, '{:08d}_disp_gt_hist.png'.format(i)))
            plt.clf()

            make_progression_video(testimgdir, "*_disp_hist.png", "progression_disp_hist.mp4")
            mse = np.mean((outputs_tof_pose['depth_map'] - target_depth)**2)
            print("Depth MSE", mse)
            
            np.save(os.path.join(testimgdir, '{:08d}_raw_depth.npy'.format(i)), outputs_tof_pose['depth_map'])
            np.save(os.path.join(testimgdir, '{:08d}_raw_depth_GT.npy'.format(i)), target_depth)
            
            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_disp.png'.format(i)), to8b(disp)
            )
            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_disp_gt.png'.format(i)), to8b(disp_gt)
            )
            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_disp_diff.png'.format(i)), to8b(disp_diff)
            )

            if 'disp_map_dynamic' in outputs_tof_pose:
                disp_dynamic = 1 - (outputs_tof_pose['depth_map_dynamic'] - self.near) / (self.far - self.near)
                disp_dynamic = cm.magma(disp_dynamic)

                imageio.imwrite(
                    os.path.join(testimgdir, '{:08d}_disp_dynamic.png'.format(i)), to8b(disp_dynamic)
                )

            if 'disp_map_static' in outputs_tof_pose:
                disp_static = 1 - (outputs_tof_pose['depth_map_static'] - self.near) / (self.far - self.near)
                disp_static = cm.magma(disp_static)

                imageio.imwrite(
                    os.path.join(testimgdir, '{:08d}_disp_static.png'.format(i)), to8b(disp_static)
                )

            make_progression_video(testimgdir, "*_disp.png", "progression_disp.mp4")
            make_progression_video(testimgdir, "*_disp_hist.png", "progression_disp_hist.mp4")
            make_progression_video(testimgdir, "*_disp_diff.png", "progression_disp_diff.mp4")
            
        if 'color' in self.images_to_log:
            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_color.png'.format(i)), cv2.cvtColor(to8b(outputs_color_pose['color_map']), cv2.COLOR_BGR2RGB)
            )
            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_color_gt.png'.format(i)), cv2.cvtColor(to8b(target_image), cv2.COLOR_BGR2RGB)
            )

            if 'color_map_dynamic' in outputs_color_pose:
                imageio.imwrite(
                    os.path.join(testimgdir, '{:08d}_color_dynamic.png'.format(i)), cv2.cvtColor(to8b(outputs_color_pose['color_map_dynamic']), cv2.COLOR_BGR2RGB)
                    )
            make_progression_video(testimgdir, "*_color.png", "progression_color.mp4")

            if 'color_map_static' in outputs_color_pose:
                imageio.imwrite(
                    os.path.join(testimgdir, '{:08d}_color_static.png'.format(i)), cv2.cvtColor(to8b(outputs_color_pose['color_map_static']), cv2.COLOR_BGR2RGB)
                    )

        tof = outputs_tof_pose['tof_map']

        tof_quads = {}
        for quad in range(4):
            if f'tofQuad_{quad}' in self.images_to_log :
                target_tofQuad = self.dataset[f'tofQuad{quad}_images'][img_i + quad][..., 0]

                tof = np.array(tof)
                
                tofQuad = tof[..., 3 + quad]
                tofQuad_norm = normalize_im_gt(tof[..., 3 + quad], target_tofQuad)
                tof_quads[quad] = tofQuad_norm

                tofQuad_gt_norm = normalize_im(target_tofQuad)

                tofQuad_diff = np.abs(tof[..., 3 + quad] - target_tofQuad) / target_tofQuad
                
                np.save(os.path.join(testimgdir, f'{i:08d}_tofQuad_{quad}.npy'), tofQuad)
                np.save(os.path.join(testimgdir, f'{i:08d}_tofQuad_{quad}_gt.npy'), target_tofQuad)

                imageio.imwrite(
                    os.path.join(testimgdir, f'{i:08d}_tofQuad_{quad}.png'), to8b(tofQuad_norm)
                )
                imageio.imwrite(
                    os.path.join(testimgdir, f'{i:08d}_tofQuad_{quad}_gt.png'), to8b(tofQuad_gt_norm)
                )
                imageio.imwrite(
                    os.path.join(testimgdir, f'{i:08d}_tofQuad_{quad}_diff.png'), to8b(tofQuad_diff)
                )

                make_progression_video(testimgdir, f"*_tofQuad_{quad}.png", f"progression_tofQuad_{quad}.mp4")
                make_progression_video(testimgdir, f"*_tofQuad_{quad}_diff.png", f"progression_tofQuad_{quad}_diff.mp4")

        if 'tof_amp' in self.images_to_log:
            tof_amp = normalize_im_gt(np.abs(tof[..., 2]), np.abs(target_tof[..., 2]))
            tof_amp_gt = normalize_im(np.abs(target_tof[..., 2]))
            tof_amp_diff = np.abs(tof[..., 2] - target_tof[..., 2]) / target_tof[..., 2]

            np.save(os.path.join(testimgdir, '{:08d}_tof_amp.npy'.format(i)), tof[..., 2])
            np.save(os.path.join(testimgdir, '{:08d}_tof_amp_gt.npy'.format(i)), target_tof[..., 2])

            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_tof_amp.png'.format(i)), to8b(tof_amp)
            )
            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_tof_amp_gt.png'.format(i)), to8b(tof_amp_gt)
            )
            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_tof_amp_diff.png'.format(i)), to8b(tof_amp_diff)
            )
            if 'tof_map_dynamic' in outputs_tof_pose:
                tof_dynamic = outputs_tof_pose['tof_map_dynamic']
                tof_amp_dynamic = normalize_im_gt(np.abs(tof_dynamic[..., 2]), np.abs(target_tof[..., 2]))
                imageio.imwrite(
                    os.path.join(testimgdir, '{:08d}_tof_amp_dynamic.png'.format(i)), to8b(tof_amp_dynamic)
                )

            if 'phase_amp' in outputs_tof_pose:
                np.save(os.path.join(testimgdir, '{:08d}_phase_amp.npy'.format(i)), outputs_tof_pose['phase_amp'])
                np.save(os.path.join(testimgdir, '{:08d}_phase_amp_raw.npy'.format(i)), outputs_tof_pose['phase_amp_raw'])

            make_progression_video(testimgdir, "*_tof_amp.png", "progression_tof_amp.mp4")
            make_progression_video(testimgdir, "*_tof_amp_diff.png", "progression_tof_amp_diff.mp4")
        
        
        if 'tof_cos' in self.images_to_log:
            tof_cos = normalize_im_gt(np.abs(tof[..., 0]), np.abs(target_tof[..., 0]))
            tof_cos_gt = normalize_im(np.abs(target_tof[..., 0]))

            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_tof_cos.png'.format(i)), to8b(tof_cos)
            )
            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_tof_cos_gt.png'.format(i)), to8b(tof_cos_gt)
            )

        if 'tof_sin' in self.images_to_log:
            tof_sin = normalize_im_gt(np.abs(tof[..., 1]), np.abs(target_tof[..., 1]))
            tof_sin_gt = normalize_im(np.abs(target_tof[..., 1]))

            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_tof_sin.png'.format(i)), to8b(tof_sin)
            )
            imageio.imwrite(
                os.path.join(testimgdir, '{:08d}_tof_sin_gt.png'.format(i)), to8b(tof_sin_gt)
            )
        
        if self.args.disocclusion and ('scene_flow' in self.images_to_log):
            occ_map = to8b(outputs_color_pose["disocclusion_map"]) # (W, H, 8)
            backward_occ = np.concatenate((occ_map[..., 0, None], occ_map[..., 1, None], occ_map[..., 2, None], occ_map[..., 3, None]), axis = 0)
            forward_occ = np.concatenate((occ_map[..., 7, None], occ_map[..., 6, None], occ_map[..., 5, None], occ_map[..., 4, None]), axis = 0)

            imageio.imwrite(
                    os.path.join(occlusionimgdir, '{:08d}_disocclusion_backward.png'.format(i)), to8b(backward_occ)
                )
            
            imageio.imwrite(
                    os.path.join(occlusionimgdir, '{:08d}_disocclusion_forward.png'.format(i)), to8b(forward_occ)
                )

            make_progression_video(occlusionimgdir, "*_disocclusion_backward.png", "progression_disocclusion_backward.mp4")
            make_progression_video(occlusionimgdir, "*_disocclusion_forward.png", "progression_disocclusion_forward.mp4")

        if 'scene_flow' in self.images_to_log: 
            scene_flow = outputs_color_pose['scene_flow_map']
            forward_scene_flow = np.array(scene_flow[..., :3])
            backwards_scene_flow = np.array(scene_flow[..., 3:])

            np.save(os.path.join(sceneflowimgdir, '{:08d}_scene_flow.npy'.format(i)), scene_flow)

            # Scene Flow Magnitude
            norm = np.expand_dims(np.linalg.norm(forward_scene_flow, axis=2), axis=2) 
            norm = (norm - norm.min()) / (norm.max() - norm.min())

            norm = np.squeeze(np.stack([norm, norm, norm], axis = 2))
            imageio.imwrite(os.path.join(sceneflowimgdir, '{:08d}_sceneflow_norm_forward.png'.format(i)), to8b(norm))
            
            norm = np.expand_dims(np.linalg.norm(backwards_scene_flow, axis=2), axis=2)
            norm = (norm - norm.min()) / (norm.max() - norm.min())

            norm = np.squeeze(np.stack([norm, norm, norm], axis = 2))
            imageio.imwrite(os.path.join(sceneflowimgdir, '{:08d}_backward_norm_forward.png'.format(i)), to8b(norm))
        
            # Reprojected Color and Depth from Neighbors
            for warp_offset in self.render_kwargs_train["warp_offsets"][1:] :
                if ((img_i + 4 * warp_offset) < 0) or ((img_i + 4 * warp_offset) >= len(i_choice)) :
                    continue

                self.render_kwargs_test['warp_offset'] = warp_offset
                warped_outputs_color = render(
                                    H=self.image_sizes['color'][1], W=self.image_sizes['color'][0],
                                    chunk=int(self.args.chunk / 32), # Hopefully prevents OOM
                                    pose=color_pose, light_pose=light_pose,
                                    ray_gen_fn=self.generate_rays['color'],
                                    image_index=fractional_img_i,
                                    **self.render_kwargs_test
                                )
                
                warped_outputs_tof = render(
                                    H=self.image_sizes['tof'][1], W=self.image_sizes['tof'][0],
                                    chunk=int(self.args.chunk / 32),
                                    pose=tof_pose, light_pose=light_pose,
                                    ray_gen_fn=self.generate_rays['tof'],
                                    image_index=fractional_img_i,
                                    **self.render_kwargs_test
                                )

                imageio.imwrite(
                    os.path.join(sceneflowimgdir, '{:08d}_color_warped_{}.png'.format(i, warp_offset)), cv2.cvtColor(to8b(warped_outputs_color['color_map']), cv2.COLOR_BGR2RGB)
                )

                warped_disp = 1 - (warped_outputs_tof['depth_map'] - self.near) / (self.far - self.near)
                warped_disp = cm.magma(warped_disp)

                imageio.imwrite(
                    os.path.join(sceneflowimgdir, '{:08d}_disp_warped_{}.png'.format(i, warp_offset)), to8b(warped_disp)
                )

                # Adding histograms for warped depth
                fig = plt.hist(np.array(np.clip(warped_outputs_tof['depth_map'], 0, 16)).flatten(), bins=500)
                plt.savefig(os.path.join(sceneflowimgdir, '{:08d}_disp_warped_{}_hist.png'.format(i, warp_offset)))
                plt.clf()

                make_progression_video(sceneflowimgdir, f"*_color_warped_{warp_offset}.png", f"progression_color_warped_{warp_offset}.mp4")
                make_progression_video(sceneflowimgdir, f"*_disp_warped_{warp_offset}.png", f"progression_disp_warped_{warp_offset}.mp4")
                make_progression_video(sceneflowimgdir, f"*_disp_warped_{warp_offset}_hist.png", f"progression_disp_warped_{warp_offset}_hist.mp4")

                self.render_kwargs_test['warp_offset'] = 0 # Reseting to default value

            # Scene Flow Projected to Optical Flow, Use OF color wheel
            width, height = self.image_sizes['color']

            source_coordinates = np.array(np.meshgrid(
                                    np.linspace(0., width - 1., width),
                                    np.linspace(0., height - 1., height),
                                    )).transpose((1, 2, 0))

            target_coordinates_forward = outputs_color_pose['forward_proj_map']
            target_coordinates_backwards = outputs_color_pose['backwards_proj_map']

            forward_flow = target_coordinates_forward - source_coordinates
            backwards_flow = target_coordinates_backwards - source_coordinates

            forward_flow_viz = flow_to_image(forward_flow, self.dataset['forward_flow'])
            backwards_flow_viz = flow_to_image(backwards_flow, self.dataset['backward_flow'])

            imageio.imwrite(os.path.join(sceneflowimgdir, '{:08d}_projected_flow_forward.png'.format(i)), forward_flow_viz)
            imageio.imwrite(os.path.join(sceneflowimgdir, '{:08d}_projected_flow_backwards.png'.format(i)), backwards_flow_viz)

            make_progression_video(sceneflowimgdir, "*_projected_flow_forward.png", "progression_projected_flow_forward.mp4")
            make_progression_video(sceneflowimgdir, "*_projected_flow_backwards.png", "progression_projected_flow_backwards.mp4")

            # Ground truth Optical flow
            imageio.imwrite(os.path.join(sceneflowimgdir, '{:08d}_projected_flow_forward_GT.png'.format(i)), flow_to_image(target_forward_flow, self.dataset['forward_flow']))
            imageio.imwrite(os.path.join(sceneflowimgdir, '{:08d}_projected_flow_backwards_GT.png'.format(i)), flow_to_image(target_backward_flow, self.dataset['backward_flow']))

            # Difference between Predicted Scene Flow Projected and GT Optical Flow
            forward_difference = flow_to_image(forward_flow - target_forward_flow, self.dataset['forward_flow'])
            backward_difference = flow_to_image(backwards_flow - target_backward_flow, self.dataset['backward_flow'])

            imageio.imwrite(os.path.join(sceneflowimgdir, '{:08d}_difference_flow_forward.png'.format(i)), forward_difference)
            imageio.imwrite(os.path.join(sceneflowimgdir, '{:08d}_difference_flow_backwards.png'.format(i)), backward_difference)

            make_progression_video(sceneflowimgdir, "*_difference_flow_forward.png", "progression_difference_flow_forward.mp4")
            make_progression_video(sceneflowimgdir, "*_difference_flow_backwards.png", "progression_difference_flow_backwards.mp4")

            # Blending weights visualization (dynamic ~= white)
            imageio.imwrite(os.path.join(sceneflowimgdir, '{:08d}_blend_weights.png'.format(i)), to8b(np.stack([np.squeeze(outputs_tof_pose['blend_weight'])] * 3, axis=2)))

            if (self.args.partial_scene_flow and (i >= self.args.pretraining_stage2_iters)) :
                target_tofQuad0 = self.dataset['tofQuad0_images'][img_i + 0][..., 0]
                target_tofQuad1 = self.dataset['tofQuad1_images'][img_i + 1][..., 0]
                target_tofQuad2 = self.dataset['tofQuad2_images'][img_i + 2][..., 0]
                target_tofQuad3 = self.dataset['tofQuad3_images'][img_i + 3][..., 0]
                
                partial_tofQuad_0 = tof_quads[0]
                target_tofQuad0 = normalize_im(target_tofQuad0)

                partial_offset = 1 
                
                partial_fractional_img_i = fractional_img_i + (partial_offset / 4.0)
                self.render_kwargs_test['warp_offset'] = fractional_img_i - partial_fractional_img_i

                partial_warped_outputs = render(
                                        H=self.image_sizes['tof'][1], W=self.image_sizes['tof'][0],
                                        chunk=int(self.args.chunk / 16), # Hopefully prevents OOM
                                        pose=tof_pose, light_pose=light_pose,
                                        ray_gen_fn=self.generate_rays['tof'],
                                        image_index=partial_fractional_img_i,
                                        **self.render_kwargs_test
                                    )
                partial_warped_tof = np.array(partial_warped_outputs['tof_map'])

                partial_tofQuad_1 = normalize_im_gt(partial_warped_tof[..., 3 + 1], target_tofQuad1)
                target_tofQuad1 = normalize_im(target_tofQuad1)

                partial_offset = 2

                partial_fractional_img_i = fractional_img_i + (partial_offset / 4.0)
                self.render_kwargs_test['warp_offset'] = fractional_img_i - partial_fractional_img_i

                partial_warped_outputs = render(
                                        H=self.image_sizes['tof'][1], W=self.image_sizes['tof'][0],
                                        chunk=int(self.args.chunk / 16), # Hopefully prevents OOM
                                        pose=tof_pose, light_pose=light_pose,
                                        ray_gen_fn=self.generate_rays['tof'],
                                        image_index=partial_fractional_img_i,
                                        **self.render_kwargs_test
                                    )
                partial_warped_tof = np.array(partial_warped_outputs['tof_map'])

                partial_tofQuad_2 = normalize_im_gt(partial_warped_tof[..., 3 + 2], target_tofQuad2)
                target_tofQuad2 = normalize_im(target_tofQuad2)

                partial_offset = 3

                partial_fractional_img_i = fractional_img_i + (partial_offset / 4.0)
                self.render_kwargs_test['warp_offset'] = fractional_img_i - partial_fractional_img_i

                partial_warped_outputs = render(
                                        H=self.image_sizes['tof'][1], W=self.image_sizes['tof'][0],
                                        chunk=int(self.args.chunk / 16), # Hopefully prevents OOM
                                        pose=tof_pose, light_pose=light_pose,
                                        ray_gen_fn=self.generate_rays['tof'],
                                        image_index=partial_fractional_img_i,
                                        **self.render_kwargs_test
                                    )
                partial_warped_tof = np.array(partial_warped_outputs['tof_map'])

                partial_tofQuad_3 = normalize_im_gt(partial_warped_tof[..., 3 + 3], target_tofQuad3)
                target_tofQuad3 = normalize_im(target_tofQuad3)

                # Visualizing our supervison formulation; Warped rays that are queried @ time t
                predicted_quads = np.concatenate([partial_tofQuad_0, partial_tofQuad_1, partial_tofQuad_2, partial_tofQuad_3], axis=0)
                gt_quads = np.concatenate([target_tofQuad0, target_tofQuad1, target_tofQuad2, target_tofQuad3], axis=0)

                imageio.imwrite(
                    os.path.join(partial_sceneflowimgdir, '{:08d}_partial_warped_quads_predicted.png'.format(i)), to8b(predicted_quads)
                )

                imageio.imwrite(
                    os.path.join(partial_sceneflowimgdir, '{:08d}_partial_warped_quads_GT.png'.format(i)), to8b(gt_quads)
                )
                make_progression_video(partial_sceneflowimgdir, "*_partial_warped_quads_predicted.png", "progression_partial_warped_quads_predicted.mp4")

        print("Image logging done.")

    def train(self):
        ## Train

        print('Begin training')
        print('TRAIN views are', self.dataloader.i_train)
        print('TEST views are', self.dataloader.i_test)
        print('VAL views are', self.dataloader.i_val)

        for i in range(self.start, self.args.N_iters + 1):
            # Update args
            self.update_args(i)

            # Model reset
            if i == self.args.model_reset_iters \
                and not self.model_reset_done:

                self.model_reset()
            
            # Calibration pretraining
            if i >= self.args.static_scene_iters \
                and not self.calibration_pretraining_done:

                self.finish_calibration_pretraining()
                
            # Evaluation
            if self.args.eval_only:
                tf.config.run_functions_eagerly(True)
                self.eval()
                tf.config.run_functions_eagerly(False)

                return

             # Compute Metrics
            if self.args.compute_metrics:
                print("computing metrics at checkpoint i", i, flush=True)
                tf.config.run_functions_eagerly(True)
                self.metric_logging(i)
                tf.config.run_functions_eagerly(False)

                return

            # Rendering
            if self.args.render_only:
                print("render only mode at checkpoint i", i, flush=True)
                tf.config.run_functions_eagerly(True)
                self.image_logging(i)
                # Rendering train views
                self.video_logging(i, mode='train')
                # Rendering spiral views
                self.video_logging(i, mode='manual_spiral')
                # Rendering freezeframe
                self.video_logging(i, mode='freezeframe')

                tf.config.run_functions_eagerly(False)

                return

            # Train step
            start_time = time.time()
            loss = 0.0
            
            # Weights
            if i % self.args.i_save == 0:
                self.save_all(i)
                
            loss, metrics, gradients = self.train_step(
                i,
                self.render_kwargs_train,
                self.args
                )

            loss_dict = loss 
            loss = loss_dict["loss"]

            time_elapsed = time.time() - start_time


            # Logging
            if i % self.args.i_print <= 1 or i < 10:
                if self.args.print_extras:
                    print("Phase offset:", self.calib_vars['phase_offset'])
                    print("Poses:", self.color_poses[0], self.color_poses[10])

                    if self.args.use_relative_poses:
                        print("Relative pose:", self.relative_pose)

                print("DC offset", self.dc_offset, self.render_kwargs_train["dc_offset"])
                print("Emitter intensity: ", self.calib_vars['emitter_intensity'])
                tf.summary.scalar("calibration/emitter_intensity", self.calib_vars['emitter_intensity'], step=i)
                
                if loss != 0.0:
                    print(self.expname, i, loss.numpy(), self.global_step.numpy(), flush=True)
                print('iter time {:.05f}'.format(time_elapsed))

                for loss_key in loss_dict.keys():
                    if loss_dict[loss_key] != 0.0: # 0.0 means that the loss is not computed on this iteration
                        tf.summary.scalar('losses/' + loss_key, loss_dict[loss_key], step=i)
                
                for metric_name, metric in metrics.items():
                    if metric != 0.0:
                        tf.summary.scalar('metrics/' + metric_name, metric, step=i)
                

            # Video Logging
            if (i % self.args.i_video == 0 and i > 0):
                tf.config.run_functions_eagerly(True)
                # Rendering train views
                self.video_logging(i, mode='train')
                # Rendering spiral views
                self.video_logging(i, mode='manual_spiral')
                # Rendering freezeframe
                self.video_logging(i, mode='freezeframe')
                tf.config.run_functions_eagerly(False)

            # Image Logging
            if (i % self.args.i_img == 0 and i > 0):
                tf.config.run_functions_eagerly(True)
                self.image_logging(i)
                tf.config.run_functions_eagerly(False)

            # Gradient Logging
            if (i % self.args.i_grad == 0) and self.args.i_grad > 0:
                pass
                # Log gradients l2 norm histograms to the tensorboard
                #for grad, var in gradients:
                    # Log L2 norm of gradients
                #    tf.summary.histogram("grad_norm_{}".format(var.name), tf.norm(grad, ord=2), step=i)


            # Increment train step
            self.global_step.assign_add(1)

        # Logging additional spiral videos at the end of training with different spiral radii
        tf.config.run_functions_eagerly(True)
        for radius in self.args.spiral_radius :
            self.video_logging(i, mode='manual_spiral', spiral_radius=radius)
        tf.config.run_functions_eagerly(False)

def main():
    if len(tf.config.list_physical_devices("GPU")) == 0 :
        exit()
    ## Parse Args
    parser = config_parser()
    args = parser.parse_args()


    if not args.resume :
       args.expname = append_time_to_experiment_name(args.expname)

    ## Random seed
    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)
    
    ## Trainer
    trainer = NeRFTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()
