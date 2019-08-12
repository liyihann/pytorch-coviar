import os
import random
import numpy as np
from coviar import get_num_frames
from coviar import load
import pickle

# each group of pictures (GoP) has 12 frames
GOP_SIZE = 12


data_root = "data/ucf101/mpeg4_videos"
_video_list = []

pickle_root = "data/ucf101/frames"

#         if self._representation == 'mv':
#             representation_idx = 1
#         elif self._representation == 'residual':
#             representation_idx = 2
#         else: # i-frame
#             representation_idx = 0

def load_list(video_list):
    # video_list: e.g. ucf101_split1_train.txt
    _video_list = []
    with open(video_list, 'r') as f:
        for line in f:
            # ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi ApplyEyeMakeup 0
            # video = *.avi
            # label = 0
            video, folder, label = line.strip().split()
            video_path = os.path.join(data_root, video[:-4] + '.mp4')
            video_name = video[video.rfind('/')+1:-4]
            num_frames = get_num_frames(video_path)
            _video_list.append((
                video_path,
                folder,
                video_name,
                int(label),
                num_frames))
            # _video_list: path(*.avi), foldername(ApplyEyeMakeup),label(0), number of frames
    with open('video_list.p','wb') as list_p:
        pickle.dump(_video_list,list_p)

# load_list("data/datalists/ucf101_split1_train.txt")


# video_list = list()
with open('video_list.p','rb') as list_p:
    video_list = pickle.load(list_p)




def get_gop_pos(frame_idx, representation):
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

'''
save_dir = "save_path/"
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)
'''

for video in video_list:
    video_path = video[0] # data/ucf101/mpeg4_videos/YoYo/v_YoYo_g25_c05.mp4    
    folder = video[1] # YoYo
    video_name = video[2] # v_YoYo_g25_c05
    label = video[3] # 100
    num_frames = video[4] # 195

    print(video_name)

    pickle_folder = os.path.join(pickle_root,folder,video_name)
    iframe_folder = os.path.join(pickle_folder,"iframe")
    mv_folder = os.path.join(pickle_folder,"mv")

    if os.path.exists(iframe_folder) is False:
        os.makedirs(iframe_folder)
    if os.path.exists(mv_folder) is False:
        os.makedirs(mv_folder)


    for i in range(num_frames):
        gop_index, gop_pos = get_gop_pos(i, 'iframe')
        iframe_img = load(video_path, gop_index, gop_pos, 0,True) # I-frame
        iframe_path = os.path.join(iframe_folder,str(gop_index)+"_"+str(gop_pos)+".p")
        with open(iframe_path,'wb') as iframe_pickle:
            pickle.dump(iframe_img,iframe_pickle)

        if i > 0:
            gop_index, gop_pos = get_gop_pos(i,'mv')
            mv_img = load(video_path, gop_index, gop_pos, 1, True) # mv
            mv_path = os.path.join(mv_folder,str(gop_index)+"_"+str(gop_pos)+".p")
            with open(mv_path,'wb')as mv_pickle:
                pickle.dump(mv_img, mv_pickle)

print("Done!")

  





