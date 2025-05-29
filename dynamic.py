import tensorflow as tf
import numpy as np
import platform
use_xla = 'Windows' in platform.system()
import time

from types import SimpleNamespace

from utils.nerf_utils import *
from utils.projection_utils import *
from utils.sampling_utils import *
from utils.temporal_utils import *
from utils.tof_utils import *
from utils.utils import *

from render import *

def convert_to_outputs_dynamic_one(
    raw, z_vals, rays_o, rays_d, pts, light_pos, near, far, visibility, args, chunk_inputs, dc_offset, dynamic=False, raw_scene_flow=None, raw_blend_weight = None, raw_disocclusion = None,
    ):
    ## Setup

    outputs = {}

    # Non-linearities for time-of-flight
    tof_nl_fn = (lambda x: tf.abs(x)) if args.use_falloff else tf.math.sigmoid

    # Distances
    dists, dists_to_cam, dists_to_light, dists_total = \
        get_dists(z_vals, rays_d, light_pos, pts)
    
    ## Geometry

    alpha, weights, transmittance = compute_geometry(
        -1, raw, z_vals, rays_d, dists_to_cam, dists, args
        )

    if visibility is None:
        visibility = transmittance[..., None]
    
    ## Color

    if 'color_map' in args.outputs:
        color_map = compute_color(0, raw, weights, args, no_over=True)
        outputs['color_map'] = color_map

    ## Time-of-flight

    if 'tof_map' in args.outputs:
        tof_map, phase_amp = compute_tof(
            3, raw, dists_to_cam, dists_total,
            weights, transmittance, visibility,
            tof_nl_fn, args,
            dc_offset=dc_offset,
            use_phasor=(args.use_phasor and (not dynamic)),
            no_over=True,
            chunk_inputs=chunk_inputs
            )
        outputs['tof_map'] = tof_map
        outputs['phase_amp'] = phase_amp

    ## Scene Flow

    if ("scene_flow_map" in args.outputs) and args.scene_flow and (raw_scene_flow is not None):
        scene_flow_map = compute_scene_flow(0, raw_scene_flow, weights, args, no_over = False)
        scene_flow_raw_map = compute_scene_flow(0, raw_scene_flow, weights, args, no_over = True)
        outputs['scene_flow_map'] = scene_flow_map
        outputs['scene_flow_raw_map'] = scene_flow_raw_map

        outputs['blend_weight_raw'] = raw_blend_weight
        outputs['blend_weight'] = tf.reduce_sum(weights[..., None] * raw_blend_weight, axis=-2)

    ## Motion Disocclusion

    if ("disocclusion_map" in args.outputs) and args.disocclusion and (raw_disocclusion is not None) :
        # Assume map order corresponds to [-1, -0.75, -0.50, -0.25, +0.25, +0.50, +0.75, +1.00]
        disocclusion_map = compute_disocclusion(0, raw_disocclusion, weights, args, no_over = False)
        disocclusion_raw_map = compute_disocclusion(0, raw_disocclusion, weights, args, no_over = True)
        outputs['disocclusion_map'] = disocclusion_map
        outputs['disocclusion_raw_map'] = disocclusion_raw_map

    return outputs, alpha, weights

# use_updated_blending function controls whether to use updated/correct blending formulation from torf++
def linear_blend(a, b, blend, use_updated_blending = True):
    if use_updated_blending :
        return a * (1 - blend) + b * blend
    else :
        return a * (1 - blend) + b

def convert_to_outputs_dynamic(
    raw, z_vals, rays_o, rays_d, pts, light_pos, near, far, visibility, args, chunk_inputs, dc_offset
    ):

    outputs = {}
    outputs['rays_d'] = rays_d
    outputs['rays_o'] = rays_o
    ## Blend
    
    if args.static_blend_weight:
        raw_scene_flow = None
        raw_disocclusion = None
        if args.scene_flow :
            if args.disocclusion :
                raw_static = raw[..., :6]
                raw_dynamic = raw[..., 7:13]
                raw_scene_flow = raw[..., 13:19]
                raw_disocclusion = raw[..., 19:27]
            else :
                raw_static = raw[..., :6]
                raw_dynamic = raw[..., 7:13]
                raw_scene_flow = raw[..., 13:19]
        else :
            raw_static = raw[..., :6]
            raw_dynamic = raw[..., 7:13]

        blend_weight = tf.math.sigmoid(raw[..., 6:7])
    else:        
        raw_scene_flow = None
        raw_disocclusion = None
        if args.scene_flow :
            if args.disocclusion :
                raw_static = raw[..., :6]
                raw_dynamic = raw[..., 6:12]
                raw_scene_flow = raw[..., 12:18]
                raw_disocclusion = raw[..., 18:26]
            else : 
                raw_static = raw[..., :6]
                raw_dynamic = raw[..., 6:12]
                raw_scene_flow = raw[..., 12:18]
        else :
            raw_static = raw[..., :6]
            raw_dynamic = raw[..., 6:12] 

        blend_weight = tf.math.sigmoid(raw[..., -1:])
        
    # Modify blending weight to use only dynamic, static, or both
    if args.static_dynamic_integration == 'dynamic' :
        blend_weight = tf.ones_like(blend_weight, dtype=tf.float32)
    elif args.static_dynamic_integration == 'static' :
        blend_weight = tf.zeros_like(blend_weight, dtype=tf.float32)
    elif args.static_dynamic_integration == 'both' :
        pass 
    else : 
        raise NotImplementedError(f'--static_dynamic_integration received unexpected input {args.static_dynamic_integration}')


    outputs_static, alpha_static, _ = convert_to_outputs_dynamic_one(
        raw_static, z_vals, rays_o, rays_d, pts, light_pos, near, far, visibility, args, chunk_inputs, dc_offset=dc_offset, dynamic=False, raw_scene_flow=None
    )

    outputs_dynamic, alpha_dynamic, _ = convert_to_outputs_dynamic_one(
        raw_dynamic, z_vals, rays_o, rays_d, pts, light_pos, near, far, visibility, args, chunk_inputs, dc_offset=dc_offset, dynamic=True, raw_scene_flow=raw_scene_flow, raw_blend_weight=blend_weight, raw_disocclusion=raw_disocclusion
    )

    alpha_static = alpha_static[..., None]
    alpha_dynamic = alpha_dynamic[..., None]
    alpha_blended = linear_blend(alpha_static, alpha_dynamic, blend_weight, args.use_quads)

    transmittance_blended = tf.math.cumprod(
        1. - alpha_blended + 1e-10, axis=-2, exclusive=True
    )
    weights_blended = alpha_blended * transmittance_blended

    def normalize_weights(weights):
        return weights / (tf.reduce_sum(weights, axis=-2, keepdims=True) + 1e-5)
    
    if args.normalize_render_weights:
        weights_blended = normalize_weights(weights_blended)

    if 'color_map' in args.outputs:
        color_weights_static = alpha_static * transmittance_blended
        color_weights_dynamic = alpha_dynamic * transmittance_blended

        if args.normalize_render_weights:
            color_weights_static = normalize_weights(color_weights_static)
            color_weights_dynamic = normalize_weights(color_weights_dynamic)

        color_blended = linear_blend(
            outputs_static['color_map'] * color_weights_static, outputs_dynamic['color_map'] * color_weights_dynamic, blend_weight, args.use_quads
            )

        color_map = tf.reduce_sum(
            color_blended,
            axis=-2
            ) 

        outputs['color_map'] = color_map

    if 'tof_map' in args.outputs:
        
        if args.normalize_render_weights:
            if args.square_transmittance:
                tof_weights_static = alpha_static * transmittance_blended * transmittance_blended
                tof_weights_dynamic = alpha_dynamic * transmittance_blended * transmittance_blended
            else:
                tof_weights_static = alpha_static * transmittance_blended
                tof_weights_dynamic = alpha_dynamic * transmittance_blended

            if args.normalize_render_weights:
                tof_weights_static = normalize_weights(tof_weights_static)
                tof_weights_dynamic = normalize_weights(tof_weights_dynamic)

            tof_blended = linear_blend(
                outputs_static['tof_map'] * tof_weights_static, outputs_dynamic['tof_map'] * tof_weights_dynamic, blend_weight, args.use_quads
            )

            tof_map = tf.reduce_sum(
                tof_blended,
                axis=-2
            ) 
        else:
            tof_blended = linear_blend(
                outputs_static['tof_map'] * alpha_static, outputs_dynamic['tof_map'] * alpha_dynamic, blend_weight, args.use_quads
                )

            if args.square_transmittance:
                tof_map = tf.reduce_sum(
                    transmittance_blended * transmittance_blended * tof_blended,
                    axis=-2
                    ) 
            else:
                tof_map = tf.reduce_sum(
                    transmittance_blended * tof_blended,
                    axis=-2
                    ) 

        outputs['tof_map'] = tof_map
        outputs['transmittance'] = transmittance_blended
        # HACK(mokunev): only using dynamic amplitude calibration factor for now
        outputs['phase_amp_raw'] = outputs_dynamic['phase_amp']
        phase_amp_map = tf.reduce_sum(
            weights_blended * outputs_dynamic['phase_amp'],
            axis=-2
        )
        outputs['phase_amp'] = phase_amp_map
        # /HACK(mokunev): only using dynamic amplitude calibration factor for now
        if hasattr(args, 'eval_model') and args.eval_model:
            outputs['tof_map_raw'] = tof_blended

    if 'disp_map' in args.outputs or 'depth_map' in args.outputs:
        depth_map = tf.reduce_sum(weights_blended * z_vals[..., None], axis=[-1, -2]) * tf.linalg.norm(rays_d, axis=-1)
        disp_map = 1. / tf.maximum(
            1e-10, depth_map / (tf.reduce_sum(weights_blended, axis=[-1, -2]) + 1e-5)
            )

        outputs['depth_map'] = depth_map
        outputs['disp_map'] = disp_map

    ## Dynamic outputs
    transmittance_dynamic = tf.math.cumprod(
        1. - alpha_dynamic + 1e-10, axis=-2, exclusive=True
    )
    weights_dynamic = alpha_dynamic * transmittance_dynamic

    if args.normalize_render_weights:
        weights_dynamic = normalize_weights(weights_dynamic)

    ## Static outputs
    transmittance_static = tf.math.cumprod(
        1. - alpha_static + 1e-10, axis=-2, exclusive=True
    )
    weights_static = alpha_static * transmittance_static
    
    if args.normalize_render_weights:
        weights_static = normalize_weights(weights_static)

    if 'color_map_dynamic' in args.outputs:
        color_map = tf.reduce_sum(
            weights_dynamic * outputs_dynamic['color_map'],
            axis=-2
            ) 
        outputs['color_map_dynamic'] = color_map

    if 'color_map_static' in args.outputs:
        color_map = tf.reduce_sum(
            weights_static * outputs_static['color_map'],
            axis=-2
            ) 
        outputs['color_map_static'] = color_map

    if 'tof_map_dynamic' in args.outputs:
        if args.square_transmittance:
            tof_dynamic_weights = transmittance_dynamic * transmittance_dynamic
            if args.normalize_render_weights:
                tof_dynamic_weights = normalize_weights(tof_dynamic_weights)
            tof_map = tf.reduce_sum(
                tof_dynamic_weights * outputs_dynamic['tof_map'],
                axis=-2
                ) 
        else:
            tof_map = tf.reduce_sum(
                weights_dynamic * outputs_dynamic['tof_map'],
                axis=-2
                ) 

        outputs['tof_map_dynamic'] = tof_map

    if 'disp_map_dynamic' in args.outputs or 'depth_map_dynamic' in args.outputs:
        depth_map = tf.reduce_sum(weights_dynamic * z_vals[..., None], axis=[-1, -2]) * tf.linalg.norm(rays_d, axis=-1)
        disp_map = 1. / tf.maximum(
            1e-10, depth_map / (tf.reduce_sum(weights_dynamic, axis=[-1, -2]) + 1e-5)
        )
        outputs['depth_map_dynamic'] = depth_map
        outputs['disp_map_dynamic'] = disp_map

    if 'disp_map_static' in args.outputs or 'depth_map_static' in args.outputs:
        depth_map = tf.reduce_sum(weights_static * z_vals[..., None], axis=[-1, -2]) * tf.linalg.norm(rays_d, axis=-1)
        disp_map = 1. / tf.maximum(
            1e-10, depth_map / (tf.reduce_sum(weights_static, axis=[-1, -2]) + 1e-5)
        )
        outputs['depth_map_static'] = depth_map
        outputs['disp_map_static'] = disp_map

    ## Other outputs

    if 'acc_map' in args.outputs:
        blend_map = tf.reduce_mean(
            -blend_weight * tf.math.log(blend_weight + 1e-8),
            axis=[-1, -2]
        )
        outputs['blend_map'] = blend_map
        
        alpha_blended_for_sparsity = normalize_weights(alpha_blended)
        #alpha_blended_for_sparsity = alpha_blended
        acc_map = tf.reduce_mean(
            -alpha_blended_for_sparsity * tf.math.log(alpha_blended_for_sparsity + 1e-8), axis=[-2]
        )
        outputs['acc_map'] = acc_map

    if 'disocclusion_map' in args.outputs:
        outputs['disocclusion_map'] = outputs_dynamic['disocclusion_map']
        outputs['disocclusion_raw_map'] = outputs_dynamic['disocclusion_raw_map']

    if 'scene_flow_map' in args.outputs:
        outputs['scene_flow_map'] = outputs_dynamic['scene_flow_map']
        outputs['scene_flow_raw_map'] = outputs_dynamic['scene_flow_raw_map']

    # Computing predicted pixel location with scene flow
    if 'forward_proj_map' in args.outputs or 'backwards_proj_map' in args.outputs :
        # Expected location in world space
        world_space_points = rays_o + rays_d * tf.expand_dims(outputs['depth_map'], axis = 1)

        # Warped in world space
        forward_warped_points = world_space_points + outputs['scene_flow_map'][..., :3]
        backwards_warped_points = world_space_points + outputs['scene_flow_map'][..., 3:]
        
        # Camera Space
        K = args.color_intrinsics
        # TODO(mokunev): figure out what's wrong with tof camera sf render
        # K = args.tof_intrinsics
        Rs_post, Ts_post = args.forward_pose[:3, :3], args.forward_pose[:3, 3:]
        Rs_pre, Ts_pre = args.backwards_pose[:3, :3], args.backwards_pose[:3, 3:]
        
        forward_warped_points = tf.expand_dims(forward_warped_points, axis = 2)
        forward_warped_points_cam = Rs_post @ forward_warped_points + Ts_post

        backwards_warped_points = tf.expand_dims(backwards_warped_points, axis = 2)
        backwards_warped_points_cam = Rs_pre @ backwards_warped_points + Ts_pre

        # Homogenous Coordinates
        forward_warped_points_cam = (K @ forward_warped_points_cam)[:, :, 0]
        backwards_warped_points_cam = (K @ backwards_warped_points_cam)[:, :, 0]

        # Pixel coordinates
        forward_proj = forward_warped_points_cam[:, :2] / (tf.math.abs(forward_warped_points_cam[:, 2:]) + 1e-8)
        backwards_proj = backwards_warped_points_cam[:, :2] / (tf.math.abs(backwards_warped_points_cam[:, 2:]) + 1e-8)

        outputs['forward_proj_map'] = forward_proj
        outputs['backwards_proj_map'] = backwards_proj

    # Blend weights
    if ('blend_weight' in outputs_dynamic) and ('blend_weight_raw' in outputs_dynamic) :
        outputs['blend_weight'] = outputs_dynamic['blend_weight']
        outputs['blend_weight_raw'] = outputs_dynamic['blend_weight_raw']
    
    ## Return
    outputs['density'] = raw_dynamic[..., -1]
    return outputs, weights_blended[..., 0]        

# @tf.function(experimental_compile=use_xla)
def render_rays_dynamic(
    chunk_inputs,
    scene_flow_raw_map = None,
    **kwargs
    ):
    # Outputs
    outputs = {}
    outputs_fine = {}

    # Extract inputs
    args = SimpleNamespace(**kwargs)

    rays_o = chunk_inputs['rays_o']
    rays_d = chunk_inputs['rays_d']
    light_pos = chunk_inputs['light_pos']
    viewdirs = chunk_inputs['viewdirs']

    near = chunk_inputs['near']
    far = chunk_inputs['far']
    N_rays = rays_o.shape[0]

    scale_fac = np.array(
        [args.scene_scale_x, args.scene_scale_y, args.scene_scale],
        )[None] / (args.far - args.near)
    
    ## Coarse model

    coarse_z_vals = coarse_samples(near, far, args)
    weights = None

    if not args.use_depth_sampling:
        # Perturb sample distances
        if args.perturb > 0.:
            coarse_z_vals = perturb_samples(coarse_z_vals)

        # Sample points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            coarse_z_vals[..., :, None]
        
        # Iteratively Warps pts from i->j based off warp_offset
        warp_offset = kwargs["warp_offset"]
        
        # This handles 4t scene flow warping
        while (((warp_offset >= 1) or (warp_offset <= -1)) and (scene_flow_raw_map is None) and args.use_quads) :
            # Forward or backward warp
            offset_sign = 1.0 if warp_offset > 0 else -1

            # Evaluate coarse model with current points
            latent_code = temporal_input(args.image_index + ((kwargs["warp_offset"] - warp_offset)), args)
            
            raw = args.network_query_fn[0](
                pts, scale_fac, viewdirs, latent_code, args.models
                )

            # Collect forward or backward scene flow
            scene_flow = raw[..., 12:15] if (offset_sign > 0) else raw[..., 15:18]

            # Warp points, query again
            pts = pts + scene_flow
            warp_offset = warp_offset - offset_sign

        # This handles 1t, 2t, 3t scene flow warping, after above
        if (((warp_offset > 0) or (warp_offset < 0)) and args.use_quads) :
            
            # Forward or backward warp
            offset_sign = 1.0 if warp_offset > 0 else -1    

            if (scene_flow_raw_map is not None) :
                scene_flow = scene_flow_raw_map[..., 0:3] if (offset_sign > 0) else scene_flow_raw_map[..., 3:6]
            else :                  
                # TODO: Hacky, intended for image_logging 
                # Evaluate coarse model with current points
                # print("This should only be called during logging!!!")
                
                latent_code = temporal_input(tf.math.floor(args.image_index + kwargs["warp_offset"]), args)

                # Fix for appropriate forward/backward flow
                if (offset_sign > 0) :
                    latent_code = temporal_input(args.image_index + kwargs["warp_offset"] - 1, args)
                else :
                    latent_code = temporal_input(args.image_index + kwargs["warp_offset"] + 1, args)

                raw = args.network_query_fn[0](
                    pts, scale_fac, viewdirs, latent_code, args.models
                    )
            
                # Collect forward or backward scene flow
                scene_flow = raw[..., 12:15] if (offset_sign > 0) else raw[..., 15:18]
            
            # Warp points
            pts = pts + abs(warp_offset) * scene_flow

        # Evaluate coarse model at time j
        latent_code = temporal_input(args.image_index + (kwargs["warp_offset"]), args)
        raw = args.network_query_fn[0](
            pts, scale_fac, viewdirs, latent_code, args.models
            )

        # Raw to outputs
        outputs, weights = convert_to_outputs_dynamic(
            raw, coarse_z_vals, rays_o, rays_d, pts, light_pos, near, far, None, args, chunk_inputs, args.dc_offset
            )
        
        # Storing cycle flow for cycle consistency loss
        if args.use_quads and kwargs['warp_offset'] == 0 and (not kwargs.get('video_logging', False)):
            # compute backwards scene flow for forward warped points @ (t+1)
            
            forward_warped_pts = pts + raw[..., 12:15]
            # Evaluate coarse model at time j + 1
            forward_latent_code = temporal_input(args.image_index + 1, args)
            forward_raw = args.network_query_fn[0](
                forward_warped_pts, scale_fac, viewdirs, forward_latent_code, args.models
                )
            outputs['cycle_flow_bw'] = forward_raw[..., 15:18] # backwards flow to original world points
            
            # compute forward scene flow for backwards warped points @ (t-1)
            
            backwards_warped_pts = pts + raw[..., 15:18]
            # Evaluate coarse model at time j - 1
            backwards_latent_code = temporal_input(args.image_index - 1, args)
            backwards_raw = args.network_query_fn[0](
                backwards_warped_pts, scale_fac, viewdirs, backwards_latent_code, args.models
                )
            outputs['cycle_flow_fw'] = backwards_raw[..., 12:15]

    ## Fine model
    # TODO: Extend/Abstract warping logic to fine network

    if args.N_importance > 0:
        # Fine sample distances
        fine_z_vals = fine_samples(coarse_z_vals, near, far, weights, chunk_inputs, args)

        if not args.use_depth_sampling:
            z_vals = tf.sort(tf.concat([coarse_z_vals, fine_z_vals], -1), -1)
        else:
            z_vals = tf.sort(fine_z_vals, -1)

        # Query points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]

        # Evaluate fine model
        latent_code = temporal_input(args.image_index, args)

        raw = args.network_query_fn[0](
            pts, scale_fac, viewdirs, latent_code, args.models, fine=True
            )

        # Raw to outputs
        outputs_fine, _ = convert_to_outputs_dynamic(
            raw, z_vals, rays_o, rays_d, pts, light_pos, near, far, None, args, chunk_inputs
            )

    # Return values from convert_to_outputs
    ret = {}
    ret['world_pts'] = pts
    ret['weights'] = weights
    ret['z_vals'] = coarse_z_vals
    if args.N_importance > 0:
        for key in outputs:
            ret[key + '0'] = outputs[key]

        for key in outputs_fine:
            ret[key] = outputs_fine[key]
    else:
        for key in outputs:
            ret[key] = outputs[key]

    # Other return values
    if args.N_importance > 0:
        if 'z_std' in args.outputs:
            ret['z_std'] = tf.math.reduce_std(z_vals, -1)

    # Check numerics
    for k in ret:
        tf.debugging.check_numerics(ret[k], 'output {}'.format(k))

    return ret