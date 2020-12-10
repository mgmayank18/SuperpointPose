import torch
import torch.optim as optim
from pose_estimation import PoseEstimation
from datasets.tum_dataloader import TUMDataloader
from unproject_reproject import unproject_loss


if __name__ == "__main__":
    train_seqs = ['rgbd_dataset_freiburg1_desk',
                    'rgbd_dataset_freiburg1_room',
                    'rgbd_dataset_freiburg3_long_office_household']
    #loader = TUMDataloader(train_seqs,'/zfsauton2/home/mayankgu/Geom/PyTorch/SuperPose/datasets/TUM_RGBD/')
    loader = TUMDataloader(train_seqs,'/usr0/yi-tinglin/SuperpointPose/datasets/TUM_RGBD/')
    max_iter = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PoseEstimation()
    #import pdb; pdb.set_trace()
    optimizer = optim.Adam(model.trunk.parameters(), lr=0.001)
    for i in range(max_iter):
        for data_dict in loader:
            optimizer.zero_grad()
            hm1, pts1, hm2, pts2 = model.forward(data_dict)
            key_point_loss = unproject_loss(pts1, hm1, hm2, data_dict, device)
            key_point_loss.backward()
            optimizer.step()
            print(key_point_loss)
             

    #import pdb; pdb.set_trace()