import numpy as np
import os
import scipy
import scipy.io

from PIL import Image

from datasets.tof_dataset import *

from utils.utils import  normalize_im_max, depth_from_tof
from utils.camera_utils import crop_mvs_input, scale_mvs_input, get_extrinsics, get_camera_params, recenter_poses

class QuadsDataset(ToFDataset):
    def __init__(
        self,
        args,
        file_endings={
            'tof': 'npy',
            'tofQuad': 'npy',
            'color': 'npy',
            'depth': 'npy',
            'cams': 'npy',
            'mask': 'jpg',
            'flow': 'npy'
            },
        ):
        # Stores indices for frames not found in dataset, Used to make occlusion masks all black here
        self.missing_frames = {
        'color' : set(),
        'tofType0' : set(),
        'tofType1' : set(),
        'tofType2' : set(),
        'tofType3' : set(),
        }
        
        super().__init__(args, file_endings)
                
            
      
    def _build_data_list(self, args):
        # Create idx map
        self.idx_map = {}
        frame_id = 0
        vid = 0

        while vid < args.total_num_views and frame_id < MAX_ITERS:
            tof_filename = self._get_tof_filename(frame_id)
            tofQuad0_filename = self._get_tofQuad_filename(frame_id, quadType = 0)
            tofQuad1_filename = self._get_tofQuad_filename(frame_id, quadType = 1)
            tofQuad2_filename = self._get_tofQuad_filename(frame_id, quadType = 2)
            tofQuad3_filename = self._get_tofQuad_filename(frame_id, quadType = 3)
            color_filename = self._get_color_filename(frame_id)
            depth_filename = self._get_depth_filename(frame_id)
            forward_flow_filename = self._get_flow_filename(frame_id, flow_direction="forward")
            backward_flow_filename = self._get_flow_filename(frame_id, flow_direction="backward")

            if os.path.isfile(tof_filename) or \
                os.path.isfile(tofQuad0_filename) or \
                os.path.isfile(tofQuad1_filename) or \
                os.path.isfile(tofQuad2_filename) or \
                os.path.isfile(tofQuad3_filename) or \
                os.path.isfile(forward_flow_filename) or \
                os.path.isfile(backward_flow_filename) or \
                os.path.isfile(color_filename) or \
                os.path.isfile(depth_filename):
                self.idx_map[vid] = frame_id
                vid += 1
            
            frame_id += 1

        # Create data list
        self.data_list = list( range(args.total_num_views) )

    # quadType = 0, 1, 2, 3 ; represents tof quad
    def _get_tofQuad_filename(self, frame_id, quadType):
        tof_filename = os.path.join(
                self.args.datadir,
                f'{self.args.scan}/tofType{quadType}/{frame_id:04d}.{self.file_endings["tofQuad"]}' 
                )
        
        return tof_filename

    def _get_flow_filename(self, frame_id, flow_direction):
        tof_filename = os.path.join(
                self.args.datadir,
                f'{self.args.scan}/{flow_direction}_flow/flow_{frame_id:04d}.{self.file_endings["flow"]}' 
                )
        
        return tof_filename

    def _build_dataset(self, args):
        # Empty dataset
        self.dataset = {
            'tof_poses': [],
            'color_poses': [],
            'tof_intrinsics': [],
            'color_intrinsics': [],
            'tof_light_poses': [],
            'color_light_poses': [],
            'tof_images': [],
            'tofQuad0_images': [],
            'tofQuad1_images': [],
            'tofQuad2_images': [],
            'tofQuad3_images': [],
            'forward_flow' : [],
            'backward_flow' : [],
            'color_images': [],
            'depth_images': [],
            'bounds': [],
        }

        # Cameras
        self._process_camera_params(args)

        # Get views
        view_ids = self.data_list

        # Get tof permutation from whatever order in the raw data to [cos, -cos, sin, -sin]
        if args.tof_permutation != "":
            self.tof_permutation = np.array([int(i) for i in args.tof_permutation.split(",")])
        elif os.path.exists(os.path.join(args.datadir, f'{args.scan}/tof_permutation.npy')):
            self.tof_permutation = np.load(os.path.join(args.datadir, f'{args.scan}/tof_permutation.npy'))
        else:
            self.tof_permutation = [0, 1, 2, 3]
        # We'll need an inverse permutation because we're always generating the quads in [cos, -cos, sin, -sin] and then permuting
        self.dataset['tof_permutation'] = self.tof_permutation
        self.dataset['tof_inverse_permutation'] = np.argsort(self.tof_permutation)

        # Build dataset

        # Expected shapes are updated based on frame 0000
        color_shape = tofQuad_shape = depth_shape = flow_shape = None 

        for i, vid in enumerate(view_ids):
            frame_id = self.idx_map[vid]

            # Important: Missing images/frames will be represented by nan arrays
            #            Training logic should not supervise on these frames

            tof_filename = self._get_tof_filename(frame_id)
            tofQuad0_filename = self._get_tofQuad_filename(frame_id, quadType=0) 
            tofQuad1_filename = self._get_tofQuad_filename(frame_id, quadType=1) 
            tofQuad2_filename = self._get_tofQuad_filename(frame_id, quadType=2) 
            tofQuad3_filename = self._get_tofQuad_filename(frame_id, quadType=3) 
            color_filename = self._get_color_filename(frame_id) 
            depth_filename = self._get_depth_filename(frame_id) 
            forward_flow_filename = self._get_flow_filename(frame_id, flow_direction="forward")
            backward_flow_filename = self._get_flow_filename(frame_id, flow_direction="backward")
            
            ## Camera params
            self.dataset['tof_intrinsics'] += [self.tof_intrinsics[vid]]
            self.dataset['tof_poses'] += [self.tof_poses[vid]]
            self.dataset['tof_light_poses'] += [self.tof_light_poses[vid]]

            self.dataset['color_intrinsics'] += [self.color_intrinsics[vid]]
            self.dataset['color_poses'] += [self.color_poses[vid]]
            self.dataset['color_light_poses'] += [self.color_light_poses[vid]]

            ## Flow
            forward_flow = self._read_flow(forward_flow_filename, shape = flow_shape)
            if not flow_shape : # Updating expected shape at frame 0000 
                flow_shape = forward_flow.shape 
            forward_flow = self._process_flow(forward_flow)
            self.dataset['forward_flow'].append(forward_flow)

            backward_flow = self._read_flow(backward_flow_filename, shape = flow_shape)
            backward_flow = self._process_flow(backward_flow)
            self.dataset['backward_flow'].append(backward_flow)

            ## TOF Quads
            tofQuad0_im = np.squeeze(self._read_tof(tofQuad0_filename, shape = tofQuad_shape))
            if not tofQuad_shape : # Updating expected shape at frame 0000 
                tofQuad_shape = tofQuad0_im.shape 

            tofQuad0_im = self._process_tof(tofQuad0_im)
            self.dataset['tofQuad0_images'].append(tofQuad0_im)

            tofQuad1_im = np.squeeze(self._read_tof(tofQuad1_filename, shape = tofQuad_shape))
            tofQuad1_im = self._process_tof(tofQuad1_im)
            self.dataset['tofQuad1_images'].append(tofQuad1_im)

            tofQuad2_im = np.squeeze(self._read_tof(tofQuad2_filename, shape = tofQuad_shape))
            tofQuad2_im = self._process_tof(tofQuad2_im)
            self.dataset['tofQuad2_images'].append(tofQuad2_im)

            tofQuad3_im = np.squeeze(self._read_tof(tofQuad3_filename, shape = tofQuad_shape))
            tofQuad3_im = self._process_tof(tofQuad3_im)
            self.dataset['tofQuad3_images'].append(tofQuad3_im)

            ## TOF
            tof_im = np.squeeze(self._read_tof(tof_filename, shape = [tofQuad_shape[0], tofQuad_shape[1], 3]))
            tof_im = self._process_tof(tof_im)
            self.dataset['tof_images'].append(tof_im)

            ## Color
            color_im = np.squeeze(self._read_color(color_filename, shape = color_shape))
            if not color_shape : # Updating expected shape at frame 0000 
                color_shape = color_im.shape 
            color_im = self._process_color(color_im)
            self.dataset['color_images'].append(color_im)

            ## Depth
            depth_im = np.squeeze(self._read_depth(depth_filename, shape = depth_shape))
            if not depth_shape : # Updating expected shape at frame 0000 
                depth_shape = depth_im.shape 
            depth_im = self._process_depth(depth_im)
            self.dataset['depth_images'].append(depth_im)

            ## Anything else (e.g. saving files)
            self._process_data_extra(args)

        # Post-process
        self._scale_dataset(args)
        self._stack_dataset(args)
        self._post_process_dataset(args)
        self._process_bounds(args)

        print("Printing indices for missing frames", self.missing_frames)

    def _scale_dataset(self, args):
        # Scale and crop
        self.dataset['forward_flow'], _ = crop_mvs_input(
                self.dataset['forward_flow'],
                self.dataset['color_intrinsics'],
                args.total_num_views,
                None,
                args.color_image_width,
                args.color_image_height,
                )
        self.dataset['forward_flow'], _ = scale_mvs_input(
                self.dataset['forward_flow'],
                self.dataset['color_intrinsics'],
                args.total_num_views,
                None,
                args.color_scale_factor,
                scale_values=True
                )
        self.dataset['backward_flow'], _ = crop_mvs_input(
                self.dataset['backward_flow'],
                self.dataset['color_intrinsics'],
                args.total_num_views,
                None,
                args.color_image_width,
                args.color_image_height
                )
        self.dataset['backward_flow'], _ = scale_mvs_input(
                self.dataset['backward_flow'],
                self.dataset['color_intrinsics'],
                args.total_num_views,
                None,
                args.color_scale_factor,
                scale_values=True
                )

        self.dataset['tof_images'], _ = crop_mvs_input(
                self.dataset['tof_images'],
                self.dataset['tof_intrinsics'],
                args.total_num_views,
                None,
                args.tof_image_width,
                args.tof_image_height,
                modify_intrinsics=True
                )
        self.dataset['tof_images'], _ = scale_mvs_input(
                self.dataset['tof_images'],
                self.dataset['tof_intrinsics'],
                args.total_num_views,
                None,
                args.tof_scale_factor,
                modify_intrinsics=True
                )
        self.dataset['tofQuad0_images'], _ = crop_mvs_input(
                self.dataset['tofQuad0_images'],
                self.dataset['tof_intrinsics'],
                args.total_num_views,
                None,
                args.tof_image_width,
                args.tof_image_height
                )
        self.dataset['tofQuad0_images'], _ = scale_mvs_input(
                self.dataset['tofQuad0_images'],
                self.dataset['tof_intrinsics'],
                args.total_num_views,
                None,
                args.tof_scale_factor
                )
        
        self.dataset['tofQuad1_images'], _ = crop_mvs_input(
                self.dataset['tofQuad1_images'],
                self.dataset['tof_intrinsics'],
                args.total_num_views,
                None,
                args.tof_image_width,
                args.tof_image_height
                )
        self.dataset['tofQuad1_images'], _ = scale_mvs_input(
                self.dataset['tofQuad1_images'],
                self.dataset['tof_intrinsics'],
                args.total_num_views,
                None,
                args.tof_scale_factor
                )

        self.dataset['tofQuad2_images'], _ = crop_mvs_input(
                self.dataset['tofQuad2_images'],
                self.dataset['tof_intrinsics'],
                args.total_num_views,
                None,
                args.tof_image_width,
                args.tof_image_height
                )
        self.dataset['tofQuad2_images'], _ = scale_mvs_input(
                self.dataset['tofQuad2_images'],
                self.dataset['tof_intrinsics'],
                args.total_num_views,
                None,
                args.tof_scale_factor
                )

        self.dataset['tofQuad3_images'], _ = crop_mvs_input(
                self.dataset['tofQuad3_images'],
                self.dataset['tof_intrinsics'],
                args.total_num_views,
                None,
                args.tof_image_width,
                args.tof_image_height
                )
        self.dataset['tofQuad3_images'], _ = scale_mvs_input(
                self.dataset['tofQuad3_images'],
                self.dataset['tof_intrinsics'],
                args.total_num_views,
                None,
                args.tof_scale_factor
                )

        self.dataset['color_images'], _ = crop_mvs_input(
                self.dataset['color_images'],
                self.dataset['color_intrinsics'],
                args.total_num_views,
                None,
                args.color_image_width,
                args.color_image_height,
                modify_intrinsics=True
                )
        self.dataset['color_images'], _ = scale_mvs_input(
                self.dataset['color_images'],
                self.dataset['color_intrinsics'],
                args.total_num_views,
                None,
                args.color_scale_factor,
                modify_intrinsics=True
                )

    def _stack_dataset(self, args):
        # Stack
        self.dataset['tof_images'] = normalize_im_max(np.stack(self.dataset['tof_images'])).astype(np.float32)

        self.dataset['tofQuad0_images'] = np.stack(self.dataset['tofQuad0_images']).astype(np.float32)
        self.dataset['tofQuad1_images'] = np.stack(self.dataset['tofQuad1_images']).astype(np.float32)
        self.dataset['tofQuad2_images'] = np.stack(self.dataset['tofQuad2_images']).astype(np.float32)
        self.dataset['tofQuad3_images'] = np.stack(self.dataset['tofQuad3_images']).astype(np.float32)

        self.dataset['forward_flow'] = np.stack(self.dataset['forward_flow']).astype(np.float32)
        self.dataset['backward_flow'] = np.stack(self.dataset['backward_flow']).astype(np.float32)

        self.dataset['color_images'], self.dataset["color_norm_factor"] = normalize_im_max(np.stack(self.dataset['color_images']), return_norm_factor = True)
        self.dataset['color_images'] =  self.dataset['color_images'].astype(np.float32)
        self.dataset['depth_images'] = np.stack(self.dataset['depth_images']).astype(np.float32)

        self.dataset['tof_intrinsics'] = np.stack(self.dataset['tof_intrinsics']).astype(np.float32)
        self.dataset['tof_poses'] = np.stack(self.dataset['tof_poses']).astype(np.float32)
        self.dataset['tof_light_poses'] = np.stack(self.dataset['tof_light_poses']).astype(np.float32)

        self.dataset['color_intrinsics'] = np.stack(self.dataset['color_intrinsics']).astype(np.float32)
        self.dataset['color_poses'] = np.stack(self.dataset['color_poses']).astype(np.float32)
        self.dataset['color_light_poses'] = np.stack(self.dataset['color_light_poses']).astype(np.float32)

    def _post_process_dataset(self, args):
        # In tof_dataset.py _post_process_dataset derives the depth from the phase offset of integrated TOF
        # Marc: I believe this is equivalent to the depth stored from our data 
        #       preprocessing script, so this key was removed
        # Scale all the quads by args.tof_values_scale_factor
        self.dataset['tofQuad0_images'] *= args.tof_values_scale_factor
        self.dataset['tofQuad1_images'] *= args.tof_values_scale_factor
        self.dataset['tofQuad2_images'] *= args.tof_values_scale_factor
        self.dataset['tofQuad3_images'] *= args.tof_values_scale_factor
        
        self._process_occlusion_masks(args)
    
    # Adds occlusion masks as the last channel for tofQuad0/1/2/3 and color; Intended for augmented quad dataset 
    def _process_occlusion_masks(self, args) :
        
        # Process occlusion masks for 'tofQuad0_images' 
    
        # Set occlusion masks to 0 by default
        occlusion_masks_shape = self.dataset['tofQuad0_images'].shape
        occlusion_masks = np.expand_dims(
                            np.zeros(occlusion_masks_shape, dtype=np.float32),
                            axis = 3)

        end = args.total_num_views - 3

        for i in range(0, end, 4) :
            # Add one b/c frame_0000 aligned with mask_0001_xxxx
            frame_id = self.idx_map[i] + 1 

            # 1st Frame
            occlusion_masks[i, :, :, :] = 1.0

            if (i+1) < end: # Check if i is on last frame
                # Warped 2nd Frame, i+1
                if not (i + 1 in self.missing_frames["tofType0"]) :
                    mask_filename = self._get_mask_filename(frame_id, frame_id + 1)
                    mask = self._read_mask(mask_filename)
                    occlusion_masks[i + 1, :, :, 0] = mask

                # Warped 3rd Frame, i+2
                if not (i + 2 in self.missing_frames["tofType0"]) :
                    mask_filename = self._get_mask_filename(frame_id, frame_id + 2)
                    mask = self._read_mask(mask_filename)
                    occlusion_masks[i + 2, :, :, 0] = mask

                # Warped 4th Frame, i+3
                if not (i + 3 in self.missing_frames["tofType0"]) :
                    mask_filename = self._get_mask_filename(frame_id, frame_id + 3)
                    mask = self._read_mask(mask_filename)
                    occlusion_masks[i + 3, :, :, 0] = mask            
        
        self.dataset['tofQuad0_images'] = np.concatenate(
                                            (np.expand_dims(self.dataset['tofQuad0_images'], axis = 3), occlusion_masks),
                                            axis = 3)
        # 'color_images'

        # Set occlusion masks to 0 by default
        occlusion_masks = np.expand_dims(
                            np.zeros(occlusion_masks_shape, dtype=np.float32),
                            axis = 3)
        
        end = args.total_num_views - 3

        for i in range(0, end, 4) :
            # Add one b/c frame_0000 aligned with mask_0001_xxxx
            frame_id = self.idx_map[i] + 1 

            # 1st Frame
            occlusion_masks[i, :, :, :] = 1.0

            if (i+1) < end : # Check if i is on last frame
                # Warped 2nd Frame, i+1
                if not (i + 1 in self.missing_frames["color"]) :
                    mask_filename = self._get_mask_filename(frame_id, frame_id + 1)
                    mask = self._read_mask(mask_filename)
                    occlusion_masks[i + 1, :, :, 0] = mask

                # Warped 3rd Frame, i+2
                if not (i + 1 in self.missing_frames["color"]) : 
                    mask_filename = self._get_mask_filename(frame_id, frame_id + 2)
                    mask = self._read_mask(mask_filename)
                    occlusion_masks[i + 2, :, :, 0] = mask

                # Warped 4th Frame, i+3
                if not (i + 1 in self.missing_frames["color"]) :
                    mask_filename = self._get_mask_filename(frame_id, frame_id + 3)
                    mask = self._read_mask(mask_filename)
                    occlusion_masks[i + 3, :, :, 0] = mask      

        self.dataset['color_images'] = np.concatenate(
                                        (self.dataset['color_images'], occlusion_masks),
                                         axis = 3)

        # 'tofQuad1_images'
    
        # Set occlusion masks to 0 by default
        occlusion_masks = np.expand_dims(
                            np.zeros(occlusion_masks_shape, dtype=np.float32),
                            axis = 3)

        end = args.total_num_views - 2

        for i in range(1, end, 4) :
            # Add one b/c frame_0000 aligned with mask_0001_xxxx
            frame_id = self.idx_map[i] + 1 

            # 1st Frame
            occlusion_masks[i, :, :, :] = 1.0

            if (i+1) < end : # Check if i is on last frame
                # Warped 2nd Frame, i+1
                if not (i + 1 in self.missing_frames["tofType1"]) :
                    mask_filename = self._get_mask_filename(frame_id, frame_id + 1)
                    mask = self._read_mask(mask_filename)
                    occlusion_masks[i + 1, :, :, 0] = mask

                # Warped 3rd Frame, i+2
                if not (i + 2 in self.missing_frames["tofType1"]) :
                    mask_filename = self._get_mask_filename(frame_id, frame_id + 2)
                    mask = self._read_mask(mask_filename)
                    occlusion_masks[i + 2, :, :, 0] = mask

                # Warped 4th Frame, i+3
                if not (i + 3 in self.missing_frames["tofType1"]) :
                    mask_filename = self._get_mask_filename(frame_id, frame_id + 3)
                    mask = self._read_mask(mask_filename)
                    occlusion_masks[i + 3, :, :, 0] = mask
     
        self.dataset['tofQuad1_images'] = np.concatenate(
                                            (np.expand_dims(self.dataset['tofQuad1_images'], axis = 3), occlusion_masks),
                                            axis = 3)

        # 'tofQuad2_images'
    
        # Set occlusion masks to 0 by default
        occlusion_masks = np.expand_dims(
                            np.zeros(occlusion_masks_shape, dtype=np.float32),
                            axis = 3)

        end = args.total_num_views - 1

        for i in range(2, end, 4) :
            # Add one b/c frame_0000 aligned with mask_0001_xxxx
            frame_id = self.idx_map[i] + 1 

            # 1st Frame
            occlusion_masks[i, :, :, :] = 1.0

            if (i+1) < end : # Check if i is on last frame
                # Warped 2nd Frame, i+1
                if not (i + 1 in self.missing_frames["tofType2"]) :
                    mask_filename = self._get_mask_filename(frame_id, frame_id + 1)
                    mask = self._read_mask(mask_filename)
                    occlusion_masks[i + 1, :, :, 0] = mask

                # Warped 3rd Frame, i+2
                if not (i + 2 in self.missing_frames["tofType2"]) :
                    mask_filename = self._get_mask_filename(frame_id, frame_id + 2)
                    mask = self._read_mask(mask_filename)
                    occlusion_masks[i + 2, :, :, 0] = mask

                # Warped 4th Frame, i+3
                if not (i + 3 in self.missing_frames["tofType2"]) :
                    mask_filename = self._get_mask_filename(frame_id, frame_id + 3)
                    mask = self._read_mask(mask_filename)
                    occlusion_masks[i + 3, :, :, 0] = mask
     
        self.dataset['tofQuad2_images'] = np.concatenate(
                                            (np.expand_dims(self.dataset['tofQuad2_images'], axis = 3), occlusion_masks),
                                            axis = 3)

        # 'tofQuad3_images'
    
        # Set occlusion masks to 0 by default
        occlusion_masks = np.expand_dims(
                            np.zeros(occlusion_masks_shape, dtype=np.float32),
                            axis = 3)

        end = args.total_num_views - 0

        for i in range(3, end, 4) :
            # Add one b/c frame_0000 aligned with mask_0001_xxxx
            frame_id = self.idx_map[i] + 1 

            # 1st Frame
            occlusion_masks[i, :, :, :] = 1.0

            if (i+1) < end : # Check if i is on last frame
                # Warped 2nd Frame, i+1
                if not (i + 1 in self.missing_frames["tofType3"]) :
                    mask_filename = self._get_mask_filename(frame_id, frame_id + 1)
                    mask = self._read_mask(mask_filename)
                    occlusion_masks[i + 1, :, :, 0] = mask

                # Warped 3rd Frame, i+2
                if not (i + 2 in self.missing_frames["tofType3"]) :
                    mask_filename = self._get_mask_filename(frame_id, frame_id + 2)
                    mask = self._read_mask(mask_filename)
                    occlusion_masks[i + 2, :, :, 0] = mask

                # Warped 4th Frame, i+3
                if not (i + 3 in self.missing_frames["tofType3"]) :
                    mask_filename = self._get_mask_filename(frame_id, frame_id + 3)
                    mask = self._read_mask(mask_filename)
                    occlusion_masks[i + 3, :, :, 0] = mask
     
        self.dataset['tofQuad3_images'] = np.concatenate(
                                            (np.expand_dims(self.dataset['tofQuad3_images'], axis = 3), occlusion_masks),
                                            axis = 3)
    
    def _get_tof_filename(self, frame_id):
        tof_filename = os.path.join(
                self.args.datadir,
                f'{self.args.scan}/synthetic_tof/{frame_id:04d}.{self.file_endings["tof"]}' 
                )
        
        return tof_filename

    def _get_tof_filename(self, frame_id):
        tof_filename = os.path.join(
                self.args.datadir,
                f'{self.args.scan}/synthetic_tof/{frame_id:04d}.{self.file_endings["tof"]}' 
                )
        
        return tof_filename

    def _get_depth_filename(self, frame_id):
        depth_directory = f'synthetic_depth'
        if self.args.gt_data_dir : # metrics for synthetic data
            depth_directory = self.args.gt_data_dir 
        
        depth_filename = os.path.join(
                self.args.datadir,
                f'{self.args.scan}/{depth_directory}/{frame_id:04d}.{self.file_endings["depth"]}' 
                )
        
        return depth_filename
    
    # start_id is original frame, end_id is interpolated
    # Ex. occ_0030_0033 represents regions that were not disoccluded when warping from 0030 -> 0033
    def _get_mask_filename(self, start_id, end_id):
        mask_filename = os.path.join(
                self.args.datadir,
                f'{self.args.scan}/occ_mask/occ_{start_id:04d}_{end_id:04d}.{self.file_endings["mask"]}' 
                )
        
        return mask_filename

    def _read_mask(self, mask_filename) :
        mask = np.array(Image.open(mask_filename))
        mask = (mask / 255.0).astype(np.float32)

        return mask

    def _read_tof(self, tof_filename, shape):
        try :
            return np.load(tof_filename)
        except OSError :
            index = int(tof_filename.split('.')[-2].split('/')[-1])
            key = tof_filename.split('/')[-2]
    
            if key.startswith("tofType") :
                self.missing_frames[key].add(index)

            return np.zeros(shape, dtype=np.float32)

    def _read_flow(self, flow_filename, shape):
        try :
            return np.load(flow_filename)
        except OSError :
            ret = np.empty(shape, dtype=np.float32)
            ret.fill(np.nan)
            return ret

    def _read_color(self, color_filename, shape):
        try :
            return np.load(color_filename)
        except OSError :
            index = int(color_filename.split('.')[-2].split('/')[-1])
            self.missing_frames['color'].add(index)

            return np.zeros(shape, dtype=np.float32)

    def _read_depth(self, depth_filename, shape):
        try :
            return np.load(depth_filename)
        except OSError :
            return np.zeros(shape, dtype=np.float32)

    def _process_flow(self, flow_im):
        return flow_im.transpose((1, 2, 0))

    def get_batch(
        self,
        i_train,
        N_rand,
        H, W,
        args,
        outputs=['tofQuad0_images', 'tofQuad1_images', 'tofQuad2_images', 'tofQuad3_images', 'color_images']
        ):
        
        return super().get_batch(i_train, N_rand, H, W, args, outputs)
