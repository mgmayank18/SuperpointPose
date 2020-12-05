from associate import *
import os
from torch.utils.data import Dataset

class TUMDataloader(Dataset):
    def __init__(self, sequences, root_dir):
        self.root_dir = root_dir
        self.sequences = []
        for sequence_name in sequences:
            folder = os.path.join(root_dir, sequence_name)
            rgb_path = os.path.join(folder, 'rgb.txt')
            depth_path = os.path.join(folder, 'depth.txt')

            rgb_list = read_file_list(rgb_path)
            depth_list = read_file_list(depth_path)

            matches_rgb_depth = dict(associate(rgb_list, depth_list, 0.0, 0.02))
            print(matches_rgb_depth)
    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

if __name__ == "__main__":
    loader = TUMDataloader(['rgbd_dataset_freiburg2_xyz'],'/zfsauton2/home/mayankgu/Geom/PyTorch/pytorch-superpoint/datasets/TUM_RGBD/')