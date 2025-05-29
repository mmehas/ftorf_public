import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def get_quads(phase, dc_offset, amp=1., quad_multiplier=1.0, perm=tf.constant([0,1,2,3])):
    quads = [quad_multiplier * (tf.math.cos(phase) + dc_offset) * amp, quad_multiplier * (-tf.math.cos(phase) + dc_offset) * amp, 
             quad_multiplier * (tf.math.sin(phase) + dc_offset) * amp, quad_multiplier * (-tf.math.sin(phase) + dc_offset) * amp]
    return tf.gather(quads, perm)

def get_phasor(phase, amp=1.):
    return tf.math.cos(phase) * amp, tf.math.sin(phase) * amp

def mul_phasors(p1_real, p1_imag, p2_real, p2_imag):
    p_real = p1_real * p2_real - p1_imag * p2_imag
    p_imag = p1_real * p2_imag + p1_imag * p2_real

    return (p_real, p_imag)

def get_phase(phasor_real, phasor_imag):
    return tf.math.atan2(phasor_imag, phasor_real)

def get_amplitude(phasor_real, phasor_imag):
    return tf.math.sqrt(
        phasor_real * phasor_real + phasor_imag * phasor_imag
        )

def add_same_phase(phasor, amp):
    phasor_real, phasor_imag, phasor_amp = phasor
    inv_phasor_amp = 1. / (phasor_amp + 1e-1)

    return mul_phasors(phasor_real, phasor_imag, amp * inv_phasor_amp + 1, 0)

def same_phase(phasor, amp):
    phasor_real, phasor_imag, phasor_amp = phasor
    inv_phasor_amp = 1. / (phasor_amp + 1e-1)

    return mul_phasors(phasor_real, phasor_imag, amp * inv_phasor_amp, 0)

def get_falloff(dists_to_light, args):
    # R-squared falloff
    if args.use_falloff:
        factor = (args.tof_multiplier / (dists_to_light * dists_to_light)) \
            * (args.depth_range * args.depth_range) / (args.falloff_range * args.falloff_range)
    else:
        factor = tf.ones_like(dists_to_light)
    
    return factor

def compute_tof(
    start_idx,
    raw,
    dists_to_light, dists_total,
    weights, transmittance, visibility,
    tof_nl_fn,
    args,
    dc_offset,
    use_phasor=False,
    no_over=False,
    chunk_inputs=None
    ):
    # R-squared falloff
    factor = get_falloff(dists_to_light, args)

    # ToF phasor
    phasor_amp = tof_nl_fn(raw[..., start_idx:start_idx+1])
    phasor_phase = dists_total * (2 * np.pi / args.depth_range)

    if hasattr(args, 'phase_offset'):
        phasor_phase += args.phase_offset

    if hasattr(args, 'emitter_intensity'):
        phasor_amp *= args.emitter_intensity
        
    if chunk_inputs is not None and args.use_phase_calib:
        phasor_phase = args.tof_phase_model(
            phasor_phase,
            tf.broadcast_to(chunk_inputs['coords'][..., None, :], phasor_phase.shape[:-1] + (2,)),
            args.models
            )

    if use_phasor:
        bias_phase = tf.math.sigmoid(
            raw[..., start_idx+1:start_idx+2]
            ) * 2 * np.pi * (args.bias_range / args.depth_range)
        phasor_phase += bias_phase
    
    # Full ToF phasor
    phasor_real, phasor_imag = \
        get_phasor(phasor_phase, phasor_amp)

    if args.use_quads:
        quads = get_quads(phasor_phase, amp=phasor_amp, dc_offset=dc_offset, quad_multiplier=args.quad_multiplier, perm=args.tof_permutation)
        tof = tf.concat(
                [phasor_real, phasor_imag, phasor_amp, quads[0], quads[1], quads[2], quads[3]],
                axis=-1
                ) * factor[..., None]
    else:
        tof = tf.concat(
            [phasor_real, phasor_imag, phasor_amp],
            axis=-1
            ) * factor[..., None]

    if not no_over:
        # Over-composited TOF
        tof_map = tf.reduce_sum(
            visibility * weights[..., None] * tof,
            axis=-2
            ) 
        
        return tof_map, phasor_amp
    else:
        return tof, phasor_amp

def extract_tof(tof):
    return (tof[..., 0:1], tof[..., 1:2], tof[..., 2:3])