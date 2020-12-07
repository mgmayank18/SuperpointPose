from models.SuperPointNet_pretrained import SuperPointNet
import torch.nn.functional as F
import torch
import cv2
import numpy as np
from datasets.tum_dataloader import TUMDataloader
from unproject_reproject import unproject_loss
import copy

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
        trunk = SuperPointNet()
        trunk = trunk.to(device)

        weights = torch.load('pretrained/superpoint_v1.pth')
        trunk.load_state_dict(weights)

        self.trunk = trunk.cuda()
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

    def point_decoder(self, semi, H, W):
        """ Converts network output to keypoint heatmap.
        """
    
        Hc, Wc = int(H / self.cell), int(W / self.cell)
        dense = F.softmax(semi, dim=1).squeeze()
        nodust = dense[:-1,:,:]

        nodust = nodust.permute(1,2,0)
        heatmap = nodust.view(Hc, Wc, self.cell, self.cell)
        #print(heatmap.size())
        heatmap = heatmap.permute(0,2,1,3)
        #print(heatmap.size())
        #import pdb; pdb.set_trace()
        #heatmap = heatmap.reshape(Hc*self.cell, Wc*self.cell)
        heatmap = heatmap.contiguous().view(Hc*self.cell, Wc*self.cell)
        
        
        ys, xs = torch.where(heatmap >= self.conf_thresh) #Location of keypoints
        pts = torch.zeros(3, len(xs))
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[ys, xs]
    
        
        return heatmap, pts
        
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

    def forward(self, gray_1, gray_2, depth_1, depth_2, rel_pose):
        H, W = gray_1.shape[0], gray_1.shape[1]

        inp1 = gray1.copy()
        inp1 = (inp1.reshape(1, H, W))
        inp1 = torch.from_numpy(inp1)
        inp1 = torch.autograd.Variable(inp1).view(1, 1, H, W).to(self.device)

        inp2 = gray2.copy()
        inp2 = (inp2.reshape(1, H, W))
        inp2 = torch.from_numpy(inp2)
        inp2 = torch.autograd.Variable(inp2).view(1, 1, H, W).to(self.device)

        semi1, desc1 = self.trunk(inp1)
        semi2, desc2 = self.trunk(inp2)
        #import pdb; pdb.set_trace()
       
        heatmap1, pts1 = self.point_decoder(semi1, H, W)
        #key_desc1 = self.get_descriptor_decoder(desc1, H, W, pts1)
        heatmap2, pts2 = self.point_decoder(semi2, H, W)
        rel_pose_I = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        unproject_loss(pts1, heatmap1, heatmap2, depth1, depth2, rel_pose, self.device)
        #unproject_loss(pts1, heatmap1, torch.clone(heatmap1), depth1, depth1, rel_pose_I, self.device)
        return heatmap1, heatmap2
        # Loss Function.
        # Consecutive Frames
        # Step 1
        # NETWORK(RGB1 (HxW), RGB2 (HxW), semi1 (HxW), semi2 (HXW), desc1 (H/8, W/8), desc2, depth1 (HxW), depth2) -> R,t
        # loss_function -> Actual R,t labels
        
        # Step 2
        # Feedback Loop
        #GT - Relative Pose between rgb1 and rgb2
    #def project_hm(self, hm, R, t):

def overlap_hm(img, hm):
    hm = hm.data.cpu().numpy().squeeze()
    fin = cv2.addWeighted(hm, 0.5, img, 0.5, 0)
    return fin

if __name__ == "__main__":
    train_seqs = ['rgbd_dataset_freiburg1_desk',
                    'rgbd_dataset_freiburg1_room',
                    'rgbd_dataset_freiburg3_long_office_household']
    #loader = TUMDataloader(train_seqs,'/zfsauton2/home/mayankgu/Geom/PyTorch/SuperPose/datasets/TUM_RGBD/')
    loader = TUMDataloader(train_seqs,'/usr0/yi-tinglin/SuperpointPose/datasets/TUM_RGBD/')
    
    #H = 120
    #W = 160
    #path1 = 'icl_snippet/250.png'
    #path2 = 'icl_snippet/254.png'
    #gray1 = read_image(path1, (H, W))
    #gray2 = read_image(path2, (H, W))
    gray1, gray2, depth1, depth2, rel_pose = loader.__getitem__(5)

    model = PoseEstimation()
    hm1, hm2 = model.forward(gray1, gray2, depth1, depth2, rel_pose)
    #print(hm1.shape, hm2.shape)

    from torchvision.utils import save_image

    save_image(hm1, 'hm1.png')
    save_image(hm2, 'hm2.png')
    save_image(torch.tensor(gray1), 'gray1.png')
    save_image(torch.tensor(gray2), 'gray2.png')
    save_image(torch.tensor(overlap_hm(gray1, hm1)), 'overlap1.png')
    save_image(torch.tensor(overlap_hm(gray2, hm2)), 'overlap2.png')
    save_image(torch.tensor(overlap_hm(hm1.data.cpu().numpy().squeeze(), hm2)), 'hm_over_hm.png')
    from unproject_reproject import unprojection_reprojection
    #unprojection_reprojection(gray1, gray2, depth1, depth2, rel_pose)
    #
    #unproject_loss(gray1, gray1, depth1, depth1, rel_pose_I)

        
