from models.SuperPointNet_pretrained import SuperPointNet
import torch.nn.functional as F
import torch
import cv2
import numpy as np

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

def verify_heatmap(heatmap, heatmap1_np):
    heatmap1 = heatmap1.data.cpu().numpy()
    eps = 1e-5
    diff = np.abs(heatmap1_np - heatmap1)
    diff = np.where(diff > eps, 1, 0)
    print(np.count_nonzero(diff))

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

    def point_decorder(self, semi, H, W):
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
        heatmap = heatmap.reshape(Hc*self.cell, Wc*self.cell)
        
        
        xs, ys = torch.where(heatmap >= self.conf_thresh)
        '''
        ptr = torch.zeros(3, len(xs))
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        
        import pdb; pdb.set_trace()
        '''
        
        return heatmap
        
    def point_decorder_np(self, semi, H, W):
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

    def forward(self, gray_1, gray_2, depth_1=None, depth_2=None):
        
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
       
        heatmap1 = self.point_decorder(semi1, H, W)
        heatmap2 = self.point_decorder(semi2, H, W)

        # Loss Function.
        # Consecutive Frames
        # Step 1
        # NETWORK(RGB1 (HxW), RGB2 (HxW), semi1 (HxW), semi2 (HXW), desc1 (H/8, W/8), desc2, depth1 (HxW), depth2) -> R,t
        # loss_function -> Actual R,t labels
        
        # Step 2
        # Feedback Loop
        #GT - Relative Pose between rgb1 and rgb2

if __name__ == "__main__":
    H = 120
    W = 160
    path1 = 'icl_snippet/250.png'
    path2 = 'icl_snippet/254.png'
    gray1 = read_image(path1, (H, W))
    gray2 = read_image(path2, (H, W))
    

    model = PoseEstimation()
    model.forward(gray1, gray2)
    




        
