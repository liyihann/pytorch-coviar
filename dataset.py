"""
Definition of PyTorch "Dataset" that iterates through compressed videos
and return compressed representations (I-frames, motion vectors, 
or residuals) for training or testing.
"""

import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data

from coviar import get_num_frames
from coviar import load
from transforms import color_aug

# each group of pictures (GoP) has 12 frames
GOP_SIZE = 12
MV_STACK_SIZE = 5

def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)



class CoviarDataSet(data.Dataset):
    def __init__(self, data_root, data_name,
                 video_list,
                 representation,
                 transform,
                 num_segments,
                 is_train,
                 accumulate):

        self._data_root = data_root # root_path
        self._data_name = data_name
        self._num_segments = num_segments # num_segments
        self._representation = representation # modality
        self._transform = transform # transform
        self._is_train = is_train # test_mode
        self._accumulate = accumulate

        self._input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

        self._load_list(video_list) # list_file, _parse_list()

    def _load_list(self, video_list):
        # video_list: e.g. ucf101_split1_train.txt
        self._video_list = []
        with open(video_list, 'r') as f:
            for line in f:
                # ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi ApplyEyeMakeup 0
                # video = *.avi
                # label = 0
                video, _, label = line.strip().split()
                video_path = os.path.join(self._data_root, video[:-4] + '.mp4')
                self._video_list.append((
                    video_path,
                    int(label),
                    get_num_frames(video_path)))
                # get_num_frames, METH_VARARGS, "Getting number of frames in a video."}
                # _video_list: path(*.avi), label(0), number of frames

        print('%d videos loaded.' % len(self._video_list))




#   seg_begin, seg_end = self.get_seg_range(num_frames, self._num_segments, seg,
#                                         represen tation=self._representation)

    def get_seg_range(n, num_segments, seg, representation):
        # n = number of frames
        # num_segments: number of segments
        # seg: in range(self._num_segments)

        if representation in ['residual', 'mv']:
            n -= 1
            # exclude the 0-th frame

        seg_size = float(n - 1) / num_segments
        seg_begin = int(np.round(seg_size * seg))
        seg_end = int(np.round(seg_size * (seg + 1)))
        if seg_end == seg_begin:
            seg_end = seg_begin + 1

        if representation in ['residual', 'mv']:
            # Exclude the 0-th frame, because it's an I-frame.
            return seg_begin + 1, seg_end + 1

        return seg_begin, seg_end

    def get_gop_pos(frame_idx, representation):
        # index for sampled single frame
        # GOP_SIZE = 12     each group of pictures (GoP) has 12 frames
        # f. Floor Division(//)
        # Divides and returns the integer value of the quotient.
        # It dumps the digits after the decimal.
        gop_index = frame_idx // GOP_SIZE
        gop_pos = frame_idx % GOP_SIZE
        if representation in ['residual', 'mv']:
            # normally, non-key frames
            # just compute as above
            if gop_pos == 0: # I-frame in the group
                gop_index -= 1 # find index of I-frame in the preceding group
                gop_pos = GOP_SIZE - 1 # use the last frame in the preceding group
        else: # I-frame. The key-frame is itself
            gop_pos = 0
        return gop_index, gop_pos
        # return group index(corresponding I-frame) and position in the group

    def _get_train_frame_index(self, num_frames, seg):
        # Compute the range of the segment.
        # for both I-frame & non I-frame
        seg_begin, seg_end = self.get_seg_range(num_frames, self._num_segments, seg,
                                                 representation=self._representation)

        # Sample one frame from the segment.

        v_frame_idx = random.randint(seg_begin, seg_end - 1)
        return self.get_gop_pos(v_frame_idx, self._representation)

    def _get_test_frame_index(self, num_frames, seg):
        if self._representation in ['mv', 'residual']:
            num_frames -= 1

        seg_size = float(num_frames - 1) / self._num_segments
        v_frame_idx = int(np.round(seg_size * (seg + 0.5)))

        if self._representation in ['mv', 'residual']:
            v_frame_idx += 1

        return self.get_gop_pos(v_frame_idx, self._representation)

    def __getitem__(self, index):

        if self._representation == 'mv':
            representation_idx = 1
        elif self._representation == 'residual':
            representation_idx = 2
        else:
            representation_idx = 0

        if self._is_train:
            video_path, label, num_frames = random.choice(self._video_list)
        else:
            video_path, label, num_frames = self._video_list[index]

        frames = []
        for seg in range(self._num_segments):
            # -----------------------------ORIGINAL_CODE-----------------------------
            # if self._is_train:
            #     gop_index, gop_pos = self._get_train_frame_index(num_frames, seg)
            # else:
            #     gop_index, gop_pos = self._get_test_frame_index(num_frames, seg)
            # # returns image of the specified frame
            # img = load(video_path, gop_index, gop_pos,
            #            representation_idx, self._accumulate)
            #
            # if img is None:
            #     print('Error: loading video %s failed.' % video_path)
            #     img = np.zeros((256, 256, 2)) if self._representation == 'mv' else np.zeros((256, 256, 3))
            # else:
            #     if self._representation == 'mv':
            #         img = clip_and_scale(img, 20)
            #         img += 128
            #         img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)
            #     elif self._representation == 'residual':
            #         img += 128
            #         img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)
            #
            # if self._representation == 'iframe':
            #     img = color_aug(img)
            #
            #     # BGR to RGB. (PyTorch uses RGB according to doc.)
            #     img = img[..., ::-1]
            #
            # frames.append(img)
            # -----------------------------ORIGINAL_CODE-----------------------------
            # -----------------------------MODIFIED_CODE-----------------------------
            # Method 1: frames_length = segnum * 5, sample 5 MV frames randomly
            # if self._representation in ['mv', 'residual']:
            #     for i in range(MV_STACK_SIZE):
            #         frames = self.process_segment_random(frames,num_frames,seg,video_path,representation_idx)
            # else:
            #     frames = self.process_segment_random(frames,num_frames,seg,video_path,representation_idx)
            # -----------------------------MODIFIED_CODE-----------------------------
            # -----------------------------MODIFIED_CODE-----------------------------
            # Method 2: frames_length = segnum * 5, sample 5 MV frames consecutively
            if self._representation in ['mv', 'residual']:
                frames = self.process_segment_consecutive(frames, num_frames, seg, video_path, representation_idx)
            else:
                frames = self.process_segment_random(frames,num_frames,seg,video_path,representation_idx)
            # -----------------------------MODIFIED_CODE-----------------------------
        # frames:   before transform:
        # RGB: num_seg(3) * height * width * num_channel(3,RGB)
        # MV: (num_seg3*stack_size5) * height * width * num_channel(2,R/G,x/y)
        frames = self._transform(frames)
        # after transform:
        # RGB frames: 1. crop each frame image  2. resize each frame image  3.  flip horizontally for each frame image
        #               the dimension, number and type of elements of frames is still the same
        # MV frames:  1. crop each frame image  2. resize each frame image, stack x-channel and y-channel together 3.  flip horizontally for each frame image
        #               the output array has one more dimension than the input array

        # RGB: num_seg(3) * height * width * num_channel(3,RGB)
        # MV: (num_seg3*stack_size5) * height * width * 2 * channels(2,RG)  ???     # a little confusing here

        frames = np.array(frames)
        # each element in frames: height * width * num_channel(3,RGB)
        # RGB frames: 3 RGB frames total for a single video
        # -----------------------------MODIFIED_CODE-----------------------------
        frame_stack = []
        frame_mv = frames
        if self._representation in ['mv', 'residual']:
            for seg in range(self._num_segments):
                frame_stack.append(frame_mv[:MV_STACK_SIZE])
                frame_mv = frame_mv[MV_STACK_SIZE:]

        np.stack([stack for stack in frame_stack], axis=1) # stacking frames

        # frame_stack: num_seg(3) * stack_size(5) * height * width * channels(2,RG)   ??

        # -----------------------------MODIFIED_CODE-----------------------------

        # transpose: switch axes of input array
        # 0->0 3->1 1->2 2->3
        # just for processing with torch ??
        # RGB: num_seg * num_channel * height * width ??
        frames = np.transpose(frames, (0, 3, 1, 2))

        # Convert image and label to torch tensors
        input = torch.from_numpy(frames).float() / 255.0
        # torch.from_numpy(ndarray) → Tensor        Creates a Tensor from a numpy.ndarray.
        # The returned tensor and ndarray share the same memory.
        # Modifications to the tensor will be reflected in the ndarray and vice versa.
        # The returned tensor is not resizable.
        if self._representation == 'iframe':
            input = (input - self._input_mean) / self._input_std
        elif self._representation == 'residual':
            input = (input - 0.5) / self._input_std
        elif self._representation == 'mv':
            input = (input - 0.5)

        return input, label

    def __len__(self):
        return len(self._video_list)

#################################################################################
    # -----------------------------MODIFIED_CODE-----------------------------
    def process_segment_random(self,frames,num_frames,seg,video_path,representation_idx):
        if self._is_train:
            gop_index, gop_pos = self._get_train_frame_index(num_frames, seg)
        else:
            gop_index, gop_pos = self._get_test_frame_index(num_frames, seg)
        # returns image of the specified frame
        img = load(video_path, gop_index, gop_pos,
                   representation_idx, self._accumulate)

        if img is None:
            print('Error: loading video %s failed.' % video_path)
            img = np.zeros((256, 256, 2)) if self._representation == 'mv' else np.zeros((256, 256, 3))
        else:
            if self._representation == 'mv':
                img = clip_and_scale(img, 20)
                img += 128
                img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)
            elif self._representation == 'residual':
                img += 128
                img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)

        if self._representation == 'iframe':
            img = color_aug(img)

            # BGR to RGB. (PyTorch uses RGB according to doc.)
            img = img[..., ::-1]

        frames.append(img)
        return frames

    def process_segment_consecutive(self,frames,num_frames,seg,video_path,representation_idx):
        if self._is_train:
            gop_index, gop_pos_list = self._get_train_frame_index_consecutive(num_frames, seg)
        else:
            gop_index, gop_pos_list = self._get_test_frame_index_consecutive(num_frames, seg)
        # returns image of the specified frame

        for gop_pos in gop_pos_list:
            img = load(video_path, gop_index, gop_pos,
                   representation_idx, self._accumulate)

            if img is None:
                print('Error: loading video %s failed.' % video_path)
                img = np.zeros((256, 256, 2)) if self._representation == 'mv' else np.zeros((256, 256, 3))
            else:
                if self._representation == 'mv':
                    img = clip_and_scale(img, 20)
                    img += 128
                    img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)
                elif self._representation == 'residual':
                    img += 128
                    img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)

            if self._representation == 'iframe':
                img = color_aug(img)
                # BGR to RGB. (PyTorch uses RGB according to doc.)
                img = img[..., ::-1]
            frames.append(img)
        return frames

    def get_gop_pos_consecutive(frame_idx, representation):
        gop_pos_list = []
        gop_index = frame_idx // GOP_SIZE
        gop_pos = frame_idx % GOP_SIZE
        if representation in ['residual', 'mv']:
            if gop_pos == 0:
                gop_index -= 1
                gop_pos = GOP_SIZE - 1
            if gop_pos <= GOP_SIZE-MV_STACK_SIZE:
                for i in range(0,MV_STACK_SIZE-1):
                    gop_pos_list.append(gop_pos+i)
            else:
                for i in range(1,MV_STACK_SIZE-(GOP_SIZE-gop_pos)):
                    gop_pos_list.append(gop_pos-i)
                gop_pos_list.reverse()
                for i in range(0,MV_STACK_SIZE-gop_pos_list-1):
                    gop_pos_list.append(gop_pos+i)
        else: # dead entrance
            gop_pos = 0
            gop_pos_list.append(gop_pos)
        return gop_index, gop_pos_list

    def _get_train_frame_index_consecutive(self, num_frames, seg):
        seg_begin, seg_end = self.get_seg_range(num_frames, self._num_segments, seg,
                                                 representation=self._representation)
        v_frame_idx = random.randint(seg_begin, seg_end - 1)
        return self.get_gop_pos_consecutive(v_frame_idx, self._representation)

    def _get_test_frame_index_consecutive(self, num_frames, seg):
        if self._representation in ['mv', 'residual']:
            num_frames -= 1
        seg_size = float(num_frames - 1) / self._num_segments
        v_frame_idx = int(np.round(seg_size * (seg + 0.5)))
        if self._representation in ['mv', 'residual']:
            v_frame_idx += 1
        return self.get_gop_pos_consecutive(v_frame_idx, self._representation)
