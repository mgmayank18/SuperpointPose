from scipy.spatial.transform import Rotation
import numpy as np
from datasets.tum_dataloader import TUMDataloader

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

def unproject_loss(xs, ys, hm1, hm2, depth1, depth2, rel_pose):
    focalLength = 525.0
    centerX = 319.5
    centerY = 239.5
    scalingFactor = 1.0

    Z_ = hm1 * depth1
    Z = Z_[xs, ys]
    X = (xs - centerX) * Z / focalLength
    Y = (ys - centerY) * Z / focalLength

    vec_org = torch.stack((X,Y,Z, torch.ones(Z.size())), dim=1) #May need to change dim to make it 4XN

    vec_transf = torch.matmul(rel_pose, vec_org)

    X1, Y1, Z1 = vec_transf[0,:], vec_transf[1,:], vec_transf[2,:]

    # Need to enforce check that Z1 is not 0
    u_1 = X1 * focalLength / (Z1 + 0.0001) + centerX
    v_1 = Y1 * focalLength / (Z1 + 0.0001) + centerY

    mask_1 = (u_1 > 640)
    mask_2 = (v_1 > 480)
    mask_3 = (u_1 < 0)
    mask_4 = (v_1 < 0)
    mask = mask_1 * mask_2 * mask_3 * mask_4

    orig_xs = xs[mask]
    orig_ys = ys[mask]

    orig_hms = hm1[orig_xs, orig_ys]
    targets = hm2[int(u_1[mask]), int(v_1[mask])] # <- Not sure if int() works on tensors.

    #Loss(orig_hms, targets)
    


if __name__ == "__main__":
    train_seqs = ['rgbd_dataset_freiburg1_desk',
                    'rgbd_dataset_freiburg1_room',
                    'rgbd_dataset_freiburg3_long_office_household']
    #loader = TUMDataloader(train_seqs,'/zfsauton2/home/mayankgu/Geom/PyTorch/SuperPose/datasets/TUM_RGBD/')
    loader = TUMDataloader(train_seqs,'/usr0/yi-tinglin/SuperpointPose/datasets/TUM_RGBD/')
    gray1, gray2, depth1, depth2, rel_pose = loader.__getitem__(5)

    unprojection_reprojection(gray1, gray2, depth1, depth2, rel_pose)
    