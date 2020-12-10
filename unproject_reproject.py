from scipy.spatial.transform import Rotation
import numpy as np
from datasets.tum_dataloader import TUMDataset
import torch

import pdb; st = pdb.set_trace
def invertRT(rel_pose):
    R = rel_pose[:3,:3]
    t = np.expand_dims(rel_pose[:,3],1)
    ret = np.hstack((R.T, -t))
    return ret

def unprojection_reprojection(img1, img2, depth1, depth2, rel_pose):
    focalLength = 525.0
    centerX = 319.5
    centerY = 239.5
    #scalingFactor = 5000.0
    scalingFactor = 1.0

    #points = np.zeros((img1.shape[0],img1.shape[1],3))
    #colors = np.zeros((img1.shape[0],img1.shape[1]))
    rotated_img = np.zeros(img1.shape)

    for u in range(img1.shape[0]):
        for v in range(img1.shape[1]):
            
            color = img1[u,v]

            Z = depth1[u,v]/scalingFactor
            X = (u - centerX) * Z / focalLength
            Y = (v - centerY) * Z / focalLength
            
            
            vec_org = np.matrix([[X],[Y],[Z],[1]])
            
            vec_transf = np.dot(rel_pose, vec_org)
            
            X1, Y1, Z1 = vec_transf[0,0], vec_transf[1,0], vec_transf[2,0]
            u_1 = X1*focalLength/Z1 + centerX
            v_1 = Y1*focalLength/Z1 + centerY
            depth_1 = Z1*scalingFactor
            #print("u,v = {},{} -> X, Y, Z = {}, {}, {}".format(u, v, X, Y, Z))
            #print("X1, Y1, Z1 = {}, {}, {} -> u1, v1 = {}, {}".format(X1, Y1, Z1, u_1, v_1))
            #print("After Rot : \n", vec_org, "\n", vec_transf)
            #print('\n', '\n', u, v, u_1, v_1, '\n')

#def visualize_img

def unproject_loss(pts, hm1, hm2, data_dict, device):
    depth1 = data_dict['depth1']
    depth2 = data_dict['depth2']
    rel_pose = data_dict['rel_pose']

    focalLengthX = torch.tensor(data_dict['fx']).to(device)
    focalLengthY = torch.tensor(data_dict['fy']).to(device)
    centerX = torch.tensor(data_dict['cx']).to(device)
    centerY = torch.tensor(data_dict['cy']).to(device)
    scalingFactor = torch.tensor(5000.0).to(device)
    
    #FR1 : 517.3	516.5	318.6	255.3
    #FR3 : 535.4	539.2	320.1	247.6

    ys = pts[0, :].type(torch.long).to(device)
    xs = pts[1, :].type(torch.long).to(device)
    depth1 = torch.from_numpy(depth1).to(device)
    rel_pose = invertRT(rel_pose)
    rel_pose = torch.from_numpy(rel_pose).to(device)
    
    Z = depth1[ys, xs] / scalingFactor
    X = (xs - centerX) * Z / focalLengthX
    Y = (ys - centerY) * Z / focalLengthY
    vec_org = torch.stack((X,Y,Z, torch.ones(Z.size()).to(device)), dim=1).type(torch.float64) #May need to change dim to make it 4XN
    vec_org = vec_org.permute(1,0)
    
    vec_transf = torch.mm(rel_pose, vec_org)
    X1, Y1, Z1 = vec_transf[0,:], vec_transf[1,:], vec_transf[2,:]
    #print(X1, Y1, Z1)

    u_1 = X1 * focalLengthX / (Z1 + 0.000000001) + centerX
    v_1 = Y1 * focalLengthY / (Z1 + 0.000000001) + centerY

    mask_1 = (u_1 < 639.5)
    mask_2 = (v_1 < 479.5)
    mask_3 = (u_1 >= 0)
    mask_4 = (v_1 >= 0)

    ys_GT, xs_GT = torch.where(hm2 >= 0.015)
    #print(min(xs_GT), max(xs_GT))
    #print(min(ys_GT), max(ys_GT))
    mask = mask_1 * mask_2 * mask_3 * mask_4

    orig_xs = xs[mask]
    orig_ys = ys[mask]
    

    orig_hms = hm1[orig_ys, orig_xs]
    targets = hm2[torch.round(v_1[mask]).type(orig_xs.dtype), torch.round(u_1[mask]).type(orig_xs.dtype)] # <- This line takes time, for some reason.


    canvas = torch.zeros(hm2.shape)
    canvas[torch.round(v_1[mask]).type(orig_xs.dtype), torch.round(u_1[mask]).type(orig_xs.dtype)] = 1
    '''
    from torchvision.utils import save_image
    from pose_estimation import overlap_hm
    save_image(canvas, 'rotatedhm.png')
    save_image(torch.tensor(overlap_hm(canvas, (hm2 > 0.015))), 'rotatedhm_overlap.png')
    '''
    #Loss(orig_hms, targets)
    loss_fuction = torch.nn.MSELoss(reduction='mean')
    loss = loss_fuction(orig_hms, targets)
  
    return loss



if __name__ == "__main__":
    train_seqs = ['rgbd_dataset_freiburg1_desk',
                    'rgbd_dataset_freiburg1_room',
                    'rgbd_dataset_freiburg3_long_office_household']
    #loader = TUMDataset(train_seqs,'/zfsauton2/home/mayankgu/Geom/PyTorch/SuperPose/datasets/TUM_RGBD/')
    loader = TUMDataset(train_seqs,'/usr0/yi-tinglin/SuperpointPose/datasets/TUM_RGBD/')
    gray1, gray2, depth1, depth2, rel_pose = loader.__getitem__(5)

    unprojection_reprojection(gray1, gray2, depth1, depth2, rel_pose)
    