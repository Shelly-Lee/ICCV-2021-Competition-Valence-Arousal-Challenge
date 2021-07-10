import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
arousal_input_dir = './arousal_loss_type:ccc_batch_size:16_length:64_cnn:resnet50_mlp:[2048, 256, 256]_L:12/log'
valence_input_dir = './valence_loss_type:ccc_batch_size:16_length:64_cnn:resnet50_mlp:[2048, 256, 256]_L:12/log'
assert os.path.exists(arousal_input_dir)
output_dir = './valence_loss_type:ccc_batch_size:16_length:64_cnn:resnet50_mlp:[2048, 256, 256]_L:12/output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
video_dict = {}
for i in range(5):
    arousal_cur_fold = os.path.join(arousal_input_dir, 'fold_{}'.format(i))
    valence_cur_fold = os.path.join(valence_input_dir, 'fold_{}'.format(i))
    video_names =  os.listdir(arousal_cur_fold)
    video_names = [file.split(".")[0] for file in video_names]
    for vid in video_names:
        arousal_df = pd.read_csv(os.path.join(arousal_cur_fold, '{}.csv'.format(vid)), header=None)
        vid = vid.replace("arousal","valence")
        valence_df = pd.read_csv(os.path.join(valence_cur_fold, '{}.csv'.format(vid)), header=None)
        predictions = np.concatenate([valence_df.values, arousal_df.values], axis=-1)
        vid_id=vid.split('valence_')[1]
        if vid_id not in video_dict.keys():
            video_dict[vid_id] = []
        video_dict[vid_id].append(predictions)
num_frames = 0
for video_name in video_dict.keys():
    preds_5_folds = np.asarray(video_dict[video_name])
    assert preds_5_folds.shape[0] == 5
    preds_mean = np.mean(preds_5_folds, axis=0)
    num_frames +=preds_mean.shape[0]
    new_df = pd.DataFrame(preds_mean)
    #new_df.to_csv(os.path.join(output_dir, '{}.csv'.format(video_name)), index=False, header=None)
    np.savetxt(os.path.join(output_dir, '{}.txt'.format(video_name)), new_df, fmt='%.4f', delimiter=',', newline='\n', header="valence,arousal", comments='')
print("num_frames in test set:{}".format(num_frames))
    
