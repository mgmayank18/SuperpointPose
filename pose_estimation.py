from models.SuperPointNet_pretrained import SuperPointNet
import torch.nn.function as F

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

    def trunk_freeze(self, freeze=True):
        """ For freezing/unfreezing trunk.
        """
        for param in self.trunk.parameters():
            param.requires_grad=freeze

    def explicit_decoder(self, semi, H, W):
        """ Converts network output to keypoint heatmap.
        """
        Hc, Wc = int(H / self.cell), int(W / self.cell)
        dense = F.softmax(semi,0)
        nodust = dense[:-1,:,:]
        nodust = nodust.permute((1,2,0))
        heatmap = nodust.view((Hc, Wc, self.cell, self.cell))
        heatmap = heatmap.permute((0,2,1,3))
        heatmap = heatmap.view((Hc*self.cell, Wc*self.cell))
        return heatmap

    #def preprocess_img()

    def binarize_heatmap(self, hm): ### If you want to get where the keypoints are in the image, could be used in loss function (?)
        return torch.gt(hm, self.conf_thresh)

    def forward(self, rgb_1, rgb_2, depth_1=None, depth_2=None):
        semi1, desc1 = self.trunk(rgb_1)
        semi2, desc2 = self.trunk(rgb_2)

        H, W = rgb_1.shape[0], rgb_1.shape[1]
        heatmap1 = self.explicit_decoder(semi1, H, W)
        heatmap2 = self.explicit_decoder(semi2, H, W)
        # Loss Function.
        # Consecutive Frames
        # Step 1
        # NETWORK(RGB1 (HxW), RGB2 (HxW), semi1 (HxW), semi2 (HXW), desc1 (H/8, W/8), desc2, depth1 (HxW), depth2) -> R,t
        # loss_function -> Actual R,t labels
        
        # Step 2
        # Feedback Loop
        #GT - Relative Pose between rgb1 and rgb2




        
