from models.SuperPointNet_pretrained import SuperPointNet
import torch.nn.functional as F
import torch
import cv2
import numpy as np
from datasets.tum_dataloader import TUMDataset
from unproject_reproject import unproject_loss
import copy
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

def read_image(impath, img_size):
    """ Read image as grayscale and resize to img_size.
    Inputs
      impath: Path to input image.
      img_size: (W, H) tuple specifying resize size.
    Returns
      grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
    grayim = cv2.imread(impath, 0)
    if grayim is None:
      raise Exception('Error reading image %s' % impath)
    # Image is resized via opencv.
    interp = cv2.INTER_AREA
    grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
    grayim = (grayim.astype('float32') / 255.)
    return grayim

def verify_buffer(buffer, buffer_np):
    buffer = buffer.data.cpu().numpy()
    if(buffer.shape != buffer_np.shape):
        print("shape does not match")
    eps = 1e-5
    diff = np.abs(buffer_np - buffer)
    #print(diff)
    diff = np.where(diff > eps, 1, 0)

    print(np.count_nonzero(diff))

def verify_buffer_torch(buffer1, buffer2):
    #buffer = buffer.data.cpu().numpy()
    if(buffer1.shape != buffer2.shape):
        print("shape does not match")
    eps = 1e-5
    diff = torch.abs(buffer1 - buffer2)
    #print(diff)
    diff = torch.where(diff > eps, 1, 0)
    print(torch.count_nonzero(diff))


class PoseEstimation():
    def __init__(self):
        """ Initialize Class
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')
        trunk = SuperPointNet()
        trunk = trunk.to(device)
        #summary(trunk, (1,640,480))
        weights = torch.load('pretrained/superpoint_v1.pth')
        trunk.load_state_dict(weights)

        self.trunk = trunk.to(device)
        self.trunk_freeze()
        self.cell = 8
        self.conf_thresh = 0.015
        self.device = device

    def trunk_freeze(self, freeze=True):
        """ For freezing/unfreezing trunk.
        """
        for param in self.trunk.parameters():
            param.requires_grad=freeze

    def get_descriptor_decoder_np(self, coarse_desc, H, W, pts):
        D = coarse_desc.shape[1]
        samp_pts = pts[:2, :]
        samp_pts[0, :] = (samp_pts[0, :] / (float(W)/2.)) - 1.
        samp_pts[1, :] = (samp_pts[1, :] / (float(H)/2.)) - 1.
        samp_pts = samp_pts.transpose(0, 1).contiguous()
        samp_pts = samp_pts.view(1, 1, -1, 2)
        samp_pts = samp_pts.float()
        samp_pts = samp_pts.cuda()
        desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
        desc = desc.data.cpu().numpy().reshape(D, -1)
        div = np.linalg.norm(desc, axis=0)
        desc /= div[np.newaxis, :]
       
        return desc

    def get_descriptor_decoder(self, coarse_desc, H, W, pts):
        """ Get the final desc
        """
        D = coarse_desc.shape[1]
        #pts2 = torch.clone(pts)
        samp_pts = pts[:2, :]
        samp_pts[0, :] = (samp_pts[0, :] / (float(W)/2.)) - 1.
        samp_pts[1, :] = (samp_pts[1, :] / (float(H)/2.)) - 1.
        samp_pts = samp_pts.transpose(0, 1).contiguous()
        samp_pts = samp_pts.view(1, 1, -1, 2)
        samp_pts = samp_pts.float()
        samp_pts = samp_pts.cuda()
        desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
        desc = desc.view(D, -1)
        div = torch.norm(desc, dim=0)
        div = torch.unsqueeze(div, 0)
        desc /= div

        #desc2 = self.get_descriptor_decoder_np(coarse_desc, H, W, pts2)
        #verify_buffer(desc, desc2)
        #import pdb; pdb.set_trace()

        return desc

    def point_decoder(self, semi, B, H, W):
        """ Converts network output to keypoint heatmap.
        """
        # semi (B, 65, H/8, W/8)
    
        Hc, Wc = int(H / self.cell), int(W / self.cell)

        # dense: (B, 65, H/8, W/8)
        dense = F.softmax(semi, dim=1)
        
        # nodust: (B, 64, H/8, W/8)
        nodust = dense[:,:-1,:,:]

        # nodust: (B, H/8, W/8, 64)
        nodust = nodust.permute(0, 2,3,1)

        # heatmap: (B, H/8, W/8, 8, 8)
        heatmap = nodust.view(B, Hc, Wc, self.cell, self.cell)
        #print(heatmap.size())
        # heatmap: (B, H/8, 8, W/8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4)
        #print(heatmap.size())
        #import pdb; pdb.set_trace()
        #heatmap = heatmap.reshape(Hc*self.cell, Wc*self.cell)
        # heatmap: (B, H, W)
        heatmap = heatmap.contiguous().view(B, Hc*self.cell, Wc*self.cell)
        
        #import pdb; pdb.set_trace()
        
        
        #ys, xs = torch.where(heatmap >= self.conf_thresh) #Location of keypoints
        
        # pts: (B, 3, N)
        #pts = np.zeros
        #pts = torch.nonzero(heatmap >= self.conf_thresh) #Location of keypoints
        #import pdb; pdb.set_trace()
        
        #pts = torch.zeros(B, 3, len(xs))
        #pts[0, :] = ys
        #pts[1, :] = xs
        #pts[2, :] = heatmap[ys, xs]
    
        
        return heatmap
        
    def point_decoder_np(self, semi, H, W):
        semi = semi.data.cpu().numpy().squeeze()
        # --- Process points.
        dense = np.exp(semi) # Softmax.
        dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.
        # Remove dustbin.
        nodust = dense[:-1, :, :]
        # Reshape to get full resolution heatmap.
        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
        #print(heatmap.shape)
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        #print(heatmap.shape)
        heatmap = np.reshape(heatmap, [Hc*self.cell, Wc*self.cell])
        return heatmap
        

    #def preprocess_img()

    def binarize_heatmap(self, hm): ### If you want to get where the keypoints are in the image, could be used in loss function (?)
        return torch.gt(hm, self.conf_thresh)


    def forward(self, gray1, gray2):
        # gray1 & gray2 (B, H, W)
        #gray1 = data_dicts["gray1"]
        #gray2 = data_dicts["gray2"]
        B, H, W = gray1.shape

        #inp1 = gray1.copy()
        #inp1 = (inp1.reshape(1, H, W))
        #inp1 = torch.from_numpy(gray1)
        inp1 = torch.autograd.Variable(gray1).view(B, 1, H, W).to(self.device)

        #inp2 = gray2.copy()
        #inp2 = (inp2.reshape(1, H, W))
        #inp2 = torch.from_numpy(gray2)
        inp2 = torch.autograd.Variable(gray2).view(B, 1, H, W).to(self.device)

        # input of the super point should be [B, 1, H, W]
        # outputs: semi (N x 65 x H/8 x W/8)  desc (N x 256 x H/8 x W/8)
        semi1, desc1 = self.trunk(inp1)
        semi2, desc2 = self.trunk(inp2)
       
        heatmap1 = self.point_decoder(semi1, B, H, W)
        heatmap2 = self.point_decoder(semi2, B, H, W)
        return heatmap1, heatmap2

def overlap_hm(img, hm, x=0.5, y=0.5):
    hm = hm.data.cpu().numpy().squeeze()
    fin = cv2.addWeighted(np.array(hm).astype(float), x, np.array(img).astype(float), y, 0)
    return fin

if __name__ == "__main__":
    train_seqs = ['rgbd_dataset_freiburg1_desk',
                    'rgbd_dataset_freiburg1_room',
                    'rgbd_dataset_freiburg3_long_office_household']


    #dataset = TUMDataset(train_seqs,'/zfsauton2/home/mayankgu/Geom/PyTorch/SuperPose/datasets/TUM_RGBD/')
    dataset = TUMDataset(train_seqs,'/usr0/yi-tinglin/SuperpointPose/datasets/TUM_RGBD/')
    #loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)
    #H = 120
    #W = 160
    #path1 = 'icl_snippet/250.png'
    #path2 = 'icl_snippet/254.png'
    #gray1 = read_image(path1, (H, W))
    #gray2 = read_image(path2, (H, W))
    
    data_dicts = dataset.__getitem__(2100)
    

    gray1 = data_dicts['gray1']
    gray2 = data_dicts['gray2']
    depth1 = data_dicts['depth1']
    depth2 = data_dicts['depth2']
    rel_pose = data_dicts['rel_pose']
    model = PoseEstimation()

    H, W = gray1.shape
    gray1_batch = gray1.copy()
    gray1_batch = gray1_batch.reshape(1, H, W)
    gray1_batch = torch.from_numpy(gray1_batch)
    gray2_batch = gray2.copy()
    gray2_batch = gray2_batch.reshape(1, H, W)
    gray2_batch = torch.from_numpy(gray2_batch)
    hm1, hm2 = model.forward(gray1_batch, gray2_batch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #unproject_loss(pts1, hm1, hm2, data_dicts, device, visualize=True)
    
    from torchvision.utils import save_image
    save_image(hm1, 'hm1.png')
    save_image(hm2, 'hm2.png')
    save_image(torch.tensor(gray1), 'gray1.png')
    save_image(torch.tensor(gray2), 'gray2.png')
    #save_image(torch.tensor(depth1), 'depth1.png')
    #save_image(torch.tensor(depth2), 'depth2.png')
    save_image(torch.tensor(overlap_hm(gray1, (hm1 > 0.015))), 'overlap1.png')
    save_image(torch.tensor(overlap_hm(gray2, (hm2 > 0.015))), 'overlap2.png')
    save_image(torch.tensor(overlap_hm(hm1.data.cpu().numpy().squeeze(), hm2)), 'hm_over_hm.png')
  