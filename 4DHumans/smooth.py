import joblib
import numpy as np
import copy
from mmhuman3d.utils.demo_utils import smooth_process
from scipy.spatial.transform import Rotation as R


file_path = "./4DHumans_output.pkl"
output_path = "./output/converted_smooth.pkl"

b = joblib.load(file_path)

num_character = 0
for fframe, data in enumerate(b.items()):
    trans = data[1]['camera'][num_character]
    #shape = data[1]['smpl'][character]['betas']
    global_orient = data[1]['smpl'][num_character]['global_orient']
    body_pose = data[1]['smpl'][num_character]['body_pose']
    final_body_pose = np.vstack([global_orient, body_pose])

    r = R.from_matrix(final_body_pose)
    body_pose_vec = r.as_rotvec() 
    trans_temp = [trans[0],trans[1],0]
    if fframe==0:
        smpl_trans =trans_temp
        smpl_pose = body_pose_vec.reshape(1,72)
    else:
        smpl_pose = np.vstack([smpl_pose,body_pose_vec.reshape(1,72)])
        smpl_trans = np.vstack([smpl_trans,trans_temp])

print('trans: ',smpl_trans.shape)
print('shape: ',smpl_pose.shape)

pose = smpl_pose # (N,72), "pose" in npz file
trans = smpl_trans # (N,3), "global_t" in npz file

smooth_type_s = 'smoothnet_windowsize8'
# smooth_type_s = 'smoothnet_windowsize16'
# smooth_type_s = 'savgol'
# smooth_type_s = 'oneeuro'
# smooth_type_s = 'guas1d'
# smooth_type_s = 'smoothnet_windowsize32'
# smooth_type_s = 'smoothnet_windowsize64'

# start from 0, the interval is 2
p0 = pose[::2]
t0 = trans[::2]
frame_num = p0.shape[0]
print(frame_num)
new_pose_0 = smooth_process(p0.reshape(frame_num,24,3), 
                            # smooth_type='smoothnet_windowsize8',
                            smooth_type=smooth_type_s,
                            cfg_base_dir='configs_cliff/_base_/post_processing/').reshape(frame_num,72)

new_trans_0 = smooth_process(t0[:, np.newaxis], 
                                # smooth_type='smoothnet_windowsize8',
                                smooth_type=smooth_type_s,
                                cfg_base_dir='configs_cliff/_base_/post_processing/').reshape(frame_num,3)

# start from 1, the interval is 2
p1 = pose[1::2]
t1 = trans[1::2]
frame_num = p1.shape[0]

new_pose_1 = smooth_process(p1.reshape(frame_num,24,3), 
                            # smooth_type='smoothnet_windowsize8',
                            smooth_type=smooth_type_s,
                            cfg_base_dir='configs_cliff/_base_/post_processing/').reshape(frame_num,72)

new_trans_1 = smooth_process(t1[:, np.newaxis], 
                                # smooth_type='smoothnet_windowsize8',
                                smooth_type=smooth_type_s,
                                cfg_base_dir='configs_cliff/_base_/post_processing/').reshape(frame_num,3)

new_pose = copy.copy(pose)
new_trans = copy.copy(trans)

new_pose[::2] = new_pose_0
new_pose[1::2] = new_pose_1

new_trans[::2] = new_trans_0
new_trans[1::2] = new_trans_1

# realimentando pparar criar o pickle
for ffnew, data_new in enumerate(b.items()):
    ## De volta de rot vec para rot matrix
    r_new = R.from_rotvec(new_pose[ffnew].reshape(24,3))
    body_pose_matrix = r_new.as_matrix() 

    global_orient_new = body_pose_matrix[0].reshape(1,3,3)
    body_pose_new = body_pose_matrix[1:]

    b[data_new[0]]['camera'][num_character] = new_trans[ffnew]
    b[data_new[0]]['smpl'][num_character]['global_orient'] = global_orient_new
    b[data_new[0]]['smpl'][num_character]['body_pose'] = body_pose_new

joblib.dump(b, output_path)
