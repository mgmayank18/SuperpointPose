from .associate import *
import os
from torch.utils.data import Dataset
import cv2
import numpy as np
from utils import get_left_in_right_pose_quat
class TUMDataloader(Dataset):
    def __init__(self, sequences, root_dir):
        self.root_dir = root_dir
        self.sequences = []
        self.len = []
        self.sorted_keys_list = []
        
        for sequence_name in sequences:
            folder = os.path.join(root_dir, sequence_name)
            rgb_path = os.path.join(folder, 'rgb.txt')
            depth_path = os.path.join(folder, 'depth.txt')
            gt_path = os.path.join(folder, 'groundtruth.txt')            

            rgb_list = read_file_list(rgb_path)
            depth_list = read_file_list(depth_path)
            gt_list = read_file_list(gt_path)

            matches_rgb_depth = dict(associate(rgb_list, depth_list, 0.0, 0.02))
            matches_rgb_pose = dict(associate(rgb_list, gt_list, 0.0, 0.2))

            self.sequences.append((folder, matches_rgb_depth, matches_rgb_pose, gt_list))
            x = list(matches_rgb_depth.keys())
            self.sorted_keys_list.append(x)
            self.len.append(len(matches_rgb_depth)-1)

    def __len__(self):
        return sum(self.len)

    def _read_image(self, folder, filename, rgbd):
            path = os.path.join(self.root_dir, folder, rgbd, filename+'.png')
            cell = 8
            input_image = cv2.imread(path)
            input_image = cv2.resize(input_image, (640, 480),interpolation=cv2.INTER_AREA)
            H, W = input_image.shape[0], input_image.shape[1]
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
            input_image = input_image.astype('float32') / 255.0
            return input_image

    def get_intrinsics(self, int):
        K1 = np.array([[]])
        K2 = np.array([[]])
        K3 = np.array([[]])
    
    def __getitem__(self, idx):
        i=0
        ind = idx
        while self.len[i] <= ind:
            print(self.len[i], idx, ind)
            ind -= self.len[i]
            i+=1
        folder, matches, gt_matches, gt_data = self.sequences[i]
        sorted_keys = self.sorted_keys_list[i]

        key_1 = sorted_keys[ind]
        key_2 = sorted_keys[ind+1]
        rgb1, rgb2, depth1, depth2  = key_1, key_2, matches[key_1], matches[key_2]
        im1 = self._read_image(folder, "%.6f"%rgb1, 'rgb')
        im2 = self._read_image(folder, "%.6f"%rgb2, 'rgb')
        d1 = self._read_image(folder, "%.6f"%depth1, 'depth')
        d2 = self._read_image(folder, "%.6f"%depth2, 'depth')
        pose1 = gt_data[gt_matches[key_1]]
        pose2 = gt_data[gt_matches[key_2]]

        R, t = get_left_in_right_pose_quat(np.array(pose2[3:]).astype(float), np.expand_dims(np.array(pose2[:3]),0).astype(float), np.array(pose1[3:]).astype(float), np.expand_dims(np.array(pose1[:3]),0).astype(float))
        
        rel_pose = np.hstack((R,t.T))
        return im1, im2, d1, d2, rel_pose

if __name__ == "__main__":
    train_seqs = ['rgbd_dataset_freiburg1_desk',
                    'rgbd_dataset_freiburg1_room',
                    'rgbd_dataset_freiburg3_long_office_household']
    #loader = TUMDataloader(train_seqs,'/zfsauton2/home/mayankgu/Geom/PyTorch/SuperPose/datasets/TUM_RGBD/')
    loader = TUMDataloader(train_seqs,'/usr0/yi-tinglin/SuperpointPose/datasets/TUM_RGBD/')
    
    a,b,c,d,e = loader.__getitem__(5)
    print(e) #RETURNS tx, ty, tz, qx, qy, qz, qw