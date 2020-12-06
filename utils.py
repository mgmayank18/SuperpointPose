import numpy as np
from numpy.linalg import inv


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

    #print(right_R, right_T)
    left_pose = get_camera_pose(left_R, left_T)
    right_pose = get_camera_pose(right_R, right_T)
    #print(left_pose)
    #print(right_pose)
    left2right = np.dot(inv(right_pose), left_pose)

    left_in_right_T = left2right[0:3, 3]
    left_in_right_R = left2right[0:3, 0:3]

    return left_in_right_R, left_in_right_T

if __name__ == "__main__":

    #left_R = np.asarray([[0.07543147, 0.61393189, -0.78574661],
    #                     [0.9970987, -0.03837025, 0.06574118],
    #                     [0.01021131, -0.78842588, -0.61504501]])

    #left_T = np.asarray([[1.3405, 0.6266, 1.6575]])

    right_R = np.asarray([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])

    right_T = np.asarray([[0, 0, 0]])

    left_R = np.asarray([[0.07543147, 0.61393189, -0.78574661],
                          [0.9970987, -0.03837025, 0.06574118],
                          [0.01021131, -0.78842588, -0.61504501]])

    left_T = np.asarray([[1.3405, 0.6266, 1.6575]])

    R, T = get_left_in_right_pose(left_R, left_T, right_R, right_T)

    print(R)
    print(T)



