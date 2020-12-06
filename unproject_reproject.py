from scipy.spatial.transform import Rotation
import numpy as np

def unprojection_reprojection(img1, img2, depth1, depth2, rel_pose):
    focalLength = 525.0
    centerX = 319.5
    centerY = 239.5
    scalingFactor = 5000.0

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
            
            #print("After Rot : \n", vec_org, "\n", vec_transf)
            #print('\n', '\n', u, v, u_1, v_1, '\n')