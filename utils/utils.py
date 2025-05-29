import numpy as np
import cv2
import tensorflow as tf
import os
import datetime

from .flow_utils import flow_to_image

def has_member(key, ns):
    return key in ns.__dict__

def is_true_safe(key, ns):
    return key in ns.__dict__ and ns.__dict__[key]

def to8b(x):
    return (255*np.clip(x, 0, 1)).astype(np.uint8)

def remove_nans(im):
    im[np.isnan(im)] = 0.0

def normalize_im_max(im, return_norm_factor = False):
    im_max = np.max(im)
    im = im / im_max
    im[np.isnan(im)] = 0.
    
    if return_norm_factor :
        return im, (1.0 / im_max)
    
    return im

def normalize_im(im):
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    im[np.isnan(im)] = 0.
    return np.clip(im, 0, 1)

def normalize_im_gt(im, im_gt):
    im = (im - np.min(im_gt)) / (np.max(im_gt) - np.min(im_gt))
    im[np.isnan(im)] = 0.
    return np.clip(im, 0, 1)

def depth_to_disparity(depth, near, far):
    depth = np.where(depth < near, near * np.ones_like(depth), depth)
    depth = np.where(depth > far, far * np.ones_like(depth), depth)
    return 1.0 / depth

def depth_from_tof(tof, depth_range, phase_offset=0.0):
    tof_phase = np.arctan2(tof[..., 1:2], tof[..., 0:1])
    tof_phase -= phase_offset
    tof_phase[tof_phase < 0] = tof_phase[tof_phase < 0] + 2 * np.pi
    return tof_phase * depth_range / (4 * np.pi)

def depth_from_quads(tof_images, depth_range, phase_offset=0.0, perm=[0,1,2,3]) -> list:
    response_real = (tof_images[..., perm[0]] - tof_images[..., perm[1]]).astype(complex)
    response_img = (tof_images[..., perm[2]] - tof_images[..., perm[3]]).astype(complex)
    response = response_real + response_img * 1j
    phase = np.angle(response)
    phase -= phase_offset
    phase[phase < 0] = phase[phase < 0] + 2 * np.pi
    return phase * depth_range / (4 * np.pi)

def tof_from_depth(depth, amp, depth_range):
    tof_phase = depth * 4 * np.pi / depth_range
    amp *= 1. / np.maximum(depth * depth, (depth_range * 0.1) * (depth_range * 0.1))

    return np.stack(
        [np.cos(tof_phase) * amp, np.sin(tof_phase) * amp, amp],
        axis=-1
        )

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def resize_all_images(images, width, height, method=cv2.INTER_AREA):
    resized_images = []

    for i in range(images.shape[0]):
        resized_images.append(cv2.resize(images[i], (width, height), interpolation=method))
    
    return np.stack(resized_images, axis=0)

# OpenCV Flow Visualization Helpers: https://github.com/opencv/opencv/blob/3.1.0/samples/python/opt_flow.py#L24-
def draw_hsv(flow):
    # flow = flow * 255 # Added by Marc
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    scaleFactor = 4 # This can be changed, was originally 4
    hsv[...,2] = np.minimum(v*scaleFactor, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def draw_hsv_video(flow):
    # if flow is a list of flows, then draw each flow in the list
    if isinstance(flow, list):
        hsv_video = [draw_hsv(f) for f in flow]
    else:
        hsv_video = [draw_hsv(flow[idx]) for idx in range(flow.shape[0])]
    return np.stack(hsv_video, axis=0)


def draw_flow_video(flows, gt_flows=None):
    if gt_flows is None : # GT Panels normalized by 1 factor
        gt_flows = flows 
    if isinstance(flows, list):
        flow_video = [flow_to_image(f, gt_flows) for f in flows]
    else:
        flow_video = [flow_to_image(flows[idx], gt_flows) for idx in range(flows.shape[0])]

    return np.stack(flow_video, axis=0) / 255.0

def tof_fill_blanks(imgs: np.array, offset: int):
    # imgs: (N, H, W, C)
    # offset: int
    # return: (N, H, W, C)
    # Fill in the blanks in the tof image with the previous frame's values
    # Non-empty images are at indices idx mod 4 == offset
    # Empty images should be filled with the previous non-empty image

    # Fill in the empty images with the previous non-empty image
    empty_indices = [idx for idx in range(imgs.shape[0]) if idx % 4 != offset]
    for idx in empty_indices:
        # Get the previous non-empty image
        prev_idx = (idx // 4) * 4 + offset
        if prev_idx < imgs.shape[0]:
            imgs[idx] = imgs[prev_idx]

    return imgs

def make_progression_video(dir, pattern, output_file):
    cmd = f"ffmpeg -framerate 10 -pattern_type glob -i '{dir}/{pattern}' -c:v libx264 -pix_fmt yuv420p -y {dir}/{output_file}"
    os.system(cmd)

def append_time_to_experiment_name(expname):
    # Get the current date and time
    current_time = datetime.datetime.now()

    # Format the date and time as a string
    time_string = current_time.strftime("%Y%m%d_%H%M%S")

    # Append the formatted time to the experiment name
    modified_expname = expname + "_" + time_string

    return modified_expname