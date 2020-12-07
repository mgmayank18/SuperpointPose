from scipy.spatial.transform import Rotation
import numpy as np
from datasets.tum_dataloader import TUMDataloader

def unprojection_reprojection(img1, img2, depth1, depth2, rel_pose):
    focalLength = 525.0
    centerX = 319.5
    centerY = 239.5
    #scalingFactor = 5000.0
    scalingFactor = 1.0

    #points = np.zeros((img1.shape[0],img1.shape[1],3))
    #colors = np.zeros((img1.shape[0],img1.shape[1]))
    rotated_img = np.zeros(img1.shape)

    print('Rel Pose', rel_pose)
    
    for u in range(img2.shape[0]):
        for v in range(img2.shape[1]):
            
            color = img2[u,v]

            Z = depth2[u,v]/scalingFactor
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

if __name__ == "__main__":
    train_seqs = ['rgbd_dataset_freiburg1_desk',
                    'rgbd_dataset_freiburg1_room',
                    'rgbd_dataset_freiburg3_long_office_household']
    #loader = TUMDataloader(train_seqs,'/zfsauton2/home/mayankgu/Geom/PyTorch/SuperPose/datasets/TUM_RGBD/')
    loader = TUMDataloader(train_seqs,'/usr0/yi-tinglin/SuperpointPose/datasets/TUM_RGBD/')
    gray1, gray2, depth1, depth2, rel_pose = loader.__getitem__(5)

    unprojection_reprojection(gray1, gray2, depth1, depth2, rel_pose)
    