import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation

def get_camera_pose(R, T):
    # Inputs: numpy arrays
    # R: 3x3 roation matrix
    # T: 3x1 translation matrix
    # Return: numpy array
    # pose: 4x4 pose matrix with last row = [0,0,0,1]
    
    # M = [R | T]
    pose = np.concatenate((R, T.T), axis=1)
    # And one row [0,0,0,1] as the last row of M
    pose = np.concatenate((pose, np.array([[0,0,0,1]])), axis=0)

    return pose

def get_left_in_right_pose(left_R, left_T, right_R, right_T):
    # Inputs: numpy arrays
    # left_R, right_R: 3x3 rotation matrix
    # left_T, right_T: 3x1 translation matrix
    #
    # Return: numpy arrays
    # relative pose: R (3x3) and T (3x1)


    #left_pose = get_camera_pose(left_R, left_T).astype(np.float)
    #right_pose = get_camera_pose(right_R, right_T).astype(np.float)

    left_in_right_R = np.dot(inv(right_R), left_R)
    #import pdb; pdb.set_trace()
    left_in_right_T = left_T - right_T

    return left_in_right_R, left_in_right_T

def get_left_in_right_pose_quat(left_q, left_T, right_q, right_T):
    # Inputs: numpy arrays
    # left_q, right_q: 4x1 quaterion matrix
    # left_T, right_T: 3x1 translation matrix
    #
    # Return: numpy arrays
    # relative pose: R (3x3) and T (3x1)

    left_R = Rotation.from_quat(left_q).as_matrix().astype(np.float)
    right_R = Rotation.from_quat(right_q).as_matrix().astype(np.float)

    left_in_right_R, left_in_right_T = get_left_in_right_pose(left_R, left_T, right_R, right_T)

    return left_in_right_R, left_in_right_T


if __name__ == "__main__":


    left_rotation = Rotation.from_matrix([[0.07543147, 0.61393189, -0.78574661],
                          [0.9970987, -0.03837025, 0.06574118],
                          [0.01021131, -0.78842588, -0.61504501]])
    
    right_rotation = Rotation.from_matrix([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]])

    left_R = left_rotation.as_matrix()
    right_R = right_rotation.as_matrix()
    right_T = np.asarray([[0, 0, 0]])
    left_T = np.asarray([[1.3405, 0.6266, 1.6575]])

    R, T = get_left_in_right_pose(left_R, left_T, right_R, right_T)

    print('Answers for input as Rotation matrix')
    print(R)
    print(T)

    left_q = left_rotation.as_quat()
    right_q = right_rotation.as_quat()
    R_q, T_q = get_left_in_right_pose_quat(right_q, right_T, left_q, left_T)

    print('Answers for input as quanterion matrix')
    print(R_q)
    print(T_q)
    print(inv(R))




