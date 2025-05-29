import tensorflow as tf
import numpy as np
from utils.tof_utils import get_amplitude

def img2mse(x, y, mask = None, occ_maps = None, occ_map_idx = None, norm='L2'):
    
    if mask is not None:
        if norm == 'L2':
            norm_err = tf.square(tf.boolean_mask(x, mask) - tf.boolean_mask(y, mask))
        elif norm == 'L1':
            norm_err = tf.abs(tf.boolean_mask(x, mask) - tf.boolean_mask(y, mask))
        else:
            raise f"Unknown norm for img2mse: {norm}"
        
        if tf.size(norm_err) == 0 :
            return 0
        else :
            if occ_maps is not None :
                occ_map = occ_maps[..., occ_map_idx]
                norm_err = occ_map[:, None] * norm_err
            return tf.reduce_mean(norm_err) 
    else:
        if norm == 'L2':
            return tf.reduce_mean(tf.square(x - y))
        elif norm == 'L1':
            return tf.reduce_mean(tf.abs(x - y))
        else:
            raise f"Unknown norm for img2mse: {norm}"

def img2mse_rel(x, y, mask = None, occ_maps = None, occ_map_idx = None, norm='L2'):
    # Implement relative MSE loss instead of an absolute one
    if mask is not None:
        normalization = tf.stop_gradient(tf.boolean_mask(y, mask))
        if norm == 'L2':
            norm_err = tf.square((tf.boolean_mask(x, mask) - tf.boolean_mask(y, mask)) / (normalization + 1e-5))
        elif norm == 'L1':
            norm_err = tf.abs((tf.boolean_mask(x, mask) - tf.boolean_mask(y, mask)) / (normalization + 1e-5))
        else:
            raise f"Unknown norm for img2mse: {norm}"
        
        if tf.size(norm_err) == 0 :
            return 0
        else :
            if occ_maps is not None :
                occ_map = occ_maps[..., occ_map_idx]
                norm_err = occ_map[:, None] * norm_err
            return tf.reduce_mean(norm_err)
    
    assert False, "Mask is required for relative MSE loss"

def img2mae(x, y):
    return tf.reduce_mean(tf.abs(x - y))

def mse2psnr(x):
    if x == 0.0:
        return tf.constant(0.0, dtype=tf.float32)
    return -10.*tf.math.log(x)/tf.math.log(10.)

def variance_weighted_loss(tof, gt, c=1.):
    tof = outputs['tof_map']

    tof_std = tof[..., -1:]
    tof = tof[..., :2]
    gt = gt[..., :2]

    mse = tf.reduce_mean(tf.square(tof - gt) / (2 * tf.square(tof_std)))
    return (mse + c * tf.reduce_mean(tf.math.log(tof_std)))

def amp_derived_loss(outputs, tof_weight):
    integrated_amp = outputs['tof_map'][..., 2:3]
    quads = outputs['tof_map'][..., 3:7]
    derived_amp = get_amplitude(quads[..., 0] - quads[..., 1], quads[..., 2] - quads[..., 3])
    amp_loss = img2mse(integrated_amp, derived_amp) * tof_weight
    return amp_loss, 0.0

def amp_gt_loss(outputs, target_tof, tof_weight):
    integrated_amp = outputs['tof_map'][..., 2:3]
    gt_amp = target_tof[..., 2:3]
    amp_loss = img2mse(integrated_amp, gt_amp) * tof_weight
    return amp_loss, 0.0

def tof_loss_variance(target_tof, outputs, tof_weight):
    img_loss = variance_weighted_loss(outputs['tof_map'], target_tof) * tof_weight
    img_loss0 = 0.0

    if 'tof_map0' in outputs:
        img_loss0 = variance_weighted_loss(outputs['tof_map0'], target_tof) * tof_weight

    return img_loss, img_loss0

def tof_mse_loss(outputs, target, weight, quad=None, use_mask=False, occ_maps = None, occ_map_idx = None, norm='L2'):
    if quad is not None:
        if use_mask : # Logic for use_quad and augmented quadDataset logic
            mask = target[:, -1]
            target = target[:, 0]
            return img2mse(outputs[..., 3 + quad], target, mask, occ_maps, occ_map_idx, norm) * weight
        else :
            return img2mse(outputs[..., 3 + quad], target[..., 3 + quad], norm=norm) * weight
    else:
        return img2mse(outputs[..., :2], target[..., :2], norm=norm) * weight

def tof_rel_mse_loss(outputs, target, weight, quad=None, use_mask=False, occ_maps = None, occ_map_idx = None, norm='L2'):
    # Relative MSE loss
    if quad is not None:
        if use_mask:
            mask = target[:, -1]
            target = target[:, 0]
            return img2mse_rel(outputs[..., 3 + quad], target, mask, occ_maps, occ_map_idx, norm) * weight
        else:
            return img2mse_rel(outputs[..., 3 + quad], target[..., 3 + quad], norm=norm) * weight
    else:
        return img2mse_rel(outputs[..., :2], target[..., :2], norm=norm) * weight


def tof_loss_default(target_tof, outputs, tof_weight, quad=None, use_mask = False,  occ_maps = None, occ_map_idx = None, mode='absolute', norm='L2'):
    if mode == 'absolute':
        img_loss = tof_mse_loss(outputs['tof_map'], target_tof, tof_weight, quad, use_mask, occ_maps, occ_map_idx, norm)
        
        img_loss0 = 0.0

        if 'tof_map0' in outputs:
            img_loss0 = tof_mse_loss(outputs['tof_map0'], target_tof, tof_weight, quad, use_mask, occ_maps, occ_map_idx, norm)
    elif mode == 'relative':
        img_loss = tof_rel_mse_loss(outputs['tof_map'], target_tof, tof_weight, quad, use_mask, occ_maps, occ_map_idx, norm)
        
        img_loss0 = 0.0

        if 'tof_map0' in outputs:
            img_loss0 = tof_rel_mse_loss(outputs['tof_map0'], target_tof, tof_weight, quad, use_mask, occ_maps, occ_map_idx, norm) 
    else:
        raise f"Unknown mode for tof loss: {mode}"
    
    return img_loss, img_loss0

def color_loss_default(target_color, outputs, color_weight, use_mask = False, occ_maps = None, occ_map_idx = None):
    mask  = None
    if use_mask:
        mask = target_color[:, -1:]
        mask = tf.squeeze(mask)
        target_color = target_color[:, :-1]
    
    img_loss = img2mse(outputs['color_map'], target_color, mask, occ_maps, occ_map_idx) * color_weight
    img_loss0 = 0.0

    if 'color_map0' in outputs:
        img_loss0 = img2mse(outputs['color_map0'], target_color, mask, occ_maps, occ_map_idx) * color_weight
    
    return img_loss, img_loss0

def disparity_loss_default(target_depth, outputs, disp_weight, near, far):
    target_disp = 1. / np.clip(target_depth, near, far)

    img_loss = img2mse(outputs['disp_map'], target_disp) * disp_weight
    img_loss0 = 0.0

    if 'disp_map0' in outputs:
        img_loss0 = img2mse(outputs['disp_map0'], target_disp) * disp_weight
    
    return img_loss, img_loss0

def depth_loss_default(target_depth, outputs, depth_weight):
    img_loss = img2mse(outputs['depth_map'], target_depth) * depth_weight
    img_loss0 = 0.0

    if 'depth_map0' in outputs:
        img_loss0 = img2mse(outputs['depth_map0'], target_depth) * depth_weight
    
    return img_loss, img_loss0

def scene_flow_loss_default(target_flow, outputs, scene_flow_weight, source_coordinates, outputs_key = 'forward_proj_map') :
    # Transform coordinates
    target_coords = source_coordinates + target_flow

    img_loss = img2mae(outputs[outputs_key], target_coords) * scene_flow_weight
    img_loss0 = 0.0

    return img_loss, img_loss0
   
def minimal_scene_flow_loss(outputs, img_i, num_frames, start_idx = 0) :    
    loss = tf.reduce_mean(tf.abs(outputs['scene_flow_raw_map']))
    return loss

def minimal_disocclusion_loss(outputs, img_i, num_frames, start_idx) :
    loss = img2mae(outputs['disocclusion_map'], 1.0)
    return loss

def blending_weight_weak_prior_loss(outputs, img_i, num_frames, weak_prior = 0.5) :
    loss = img2mae(outputs['blend_weight'], weak_prior)
    return loss

# kinetic
def temporal_scene_flow_smoothness_loss(outputs, img_i, num_frames, start_idx = 0):
    # t0 and t(n-1) should ignore temporal sf regularization
    if ((img_i == start_idx) or (img_i == (num_frames - 4))) :
        return 0.0

    xyz = outputs['world_pts']
    n = xyz.shape[1]

    xyz_fw = xyz + outputs['scene_flow_raw_map'][... ,:3]
    xyz_bw = xyz + outputs['scene_flow_raw_map'][... , 3:]
    
    truncated_n = int(n * 0.9)

    xyz_close = xyz[..., :truncated_n, :]
    xyz_fw_close = xyz_fw[..., :truncated_n, :]
    xyz_bw_close = xyz_bw[..., :truncated_n, :]

    sf_w_ref2post = xyz_fw_close - xyz_close
    sf_w_prev2ref = xyz_close - xyz_bw_close
    
    loss = 0.5 * tf.reduce_mean(tf.square(sf_w_ref2post - sf_w_prev2ref))
    return loss

def spatial_scene_flow_smoothness_loss(outputs, img_i, num_frames, start_idx = 0) :
    xyz = outputs['world_pts']
    n = xyz.shape[1]

    xyz_fw = xyz + outputs['scene_flow_raw_map'][... ,:3]
    xyz_bw = xyz + outputs['scene_flow_raw_map'][... , 3:]

    truncated_n = int(n * 0.95)

    xyz_close = xyz[..., :truncated_n, :]
    xyz_fw_close = xyz_fw[..., :truncated_n, :]
    xyz_bw_close = xyz_bw[..., :truncated_n, :]
    
    sf_w_ref2post = xyz_fw_close - xyz_close
    sf_w_ref2prev = xyz_bw_close - xyz_close

    loss0 = tf.reduce_mean(tf.abs(sf_w_ref2post[..., :-1, :] - sf_w_ref2post[..., 1:, :]))
    loss1 = tf.reduce_mean(tf.abs(sf_w_ref2prev[..., :-1, :] - sf_w_ref2prev[..., 1:, :]))

    # spatial sf regularization edge cases, in these cases the forward or backward sf is unsupervised
    if (img_i == start_idx) :
        loss1 = 0.0
    if (img_i == (num_frames - 4)) :
        loss0 = 0.0

    loss = loss0 + loss1
            
    return loss

def cycle_consistency_loss(outputs, img_i, num_frames, bw_disocclusion, fw_disocclusion, start_idx = 0):
    # forwards cycle
    raw_scene_flow_fw = outputs['scene_flow_raw_map'][... ,:3]
   
    # backwards cycle
    raw_scene_flow_bw =  outputs['scene_flow_raw_map'][... , 3:] 

    if (bw_disocclusion is None) and (fw_disocclusion is None) :
        loss0 = tf.reduce_mean(tf.abs(raw_scene_flow_fw + outputs['cycle_flow_bw']))
        loss1 = tf.reduce_mean(tf.abs(raw_scene_flow_bw + outputs['cycle_flow_fw']))
    else :
        loss0 = tf.reduce_mean(fw_disocclusion[:, :, None] * tf.abs(raw_scene_flow_fw + outputs['cycle_flow_bw']))
        loss1 = tf.reduce_mean(bw_disocclusion[:, :, None] * tf.abs(raw_scene_flow_bw + outputs['cycle_flow_fw']))

    if (img_i == start_idx) :
        loss1 = 0.0
    if (img_i == (num_frames - 4)) :
        loss0 = 0.0

    loss = loss0 + loss1

    return loss

def empty_space_loss(outputs, start_idx = 0):
    loss = tf.reduce_mean(tf.abs(outputs['acc_map']))

    if 'acc_map0' in outputs:
        loss += tf.reduce_mean(tf.abs(outputs['acc_map0']))

    return loss

def make_pose_loss(model, key, start_idx = 0):
    def loss_fn(_):
        return tf.reduce_mean(tf.square(
            tf.abs(model.poses[key][1:] - model.poses[key][:-1])
        ))
    
    return loss_fn

def blend_weight_entropy_loss(outputs, img_i, num_frames, start_idx) :
    return tf.reduce_mean(-outputs['blend_weight_raw'] * tf.math.log(outputs['blend_weight_raw']+ 1e-8))

# From DyBLuRF (https://kaist-viclab.github.io/dyblurf-site/)
# In our codebase, 0 = static, 1 = dynamic, so I modified to log(blend_weight) instead of -1 * log(blend_weight)
def static_blend_weight_loss(outputs, img_i, num_frames, start_idx) :
    return tf.reduce_mean(tf.math.log(outputs['blend_weight_raw']+ 1e-8))
