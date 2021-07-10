import os
import cv2

subs=os.listdir('./valence_loss_type:ccc_batch_size:16_length:64_cnn:resnet50_mlp:[2048, 256, 256]_L:12/output')
subs.sort()
video=os.listdir('../../AffWild2_test/VA_Set/videos/Test_Set/')

for sub in subs:
  f = open('./valence_loss_type:ccc_batch_size:16_length:64_cnn:resnet50_mlp:[2048, 256, 256]_L:12/output/'+sub, 'a+')
  number_of_lines=0
  last_line=''
  f.seek(0)
  print(sub)
  sub=sub.split('_left')[0]
  sub=sub.split('_right')[0]
  sub=sub.split('.')[0]
  name=[x for x in video if x.split('.')[0]==sub]
  print(name[0])
  for line in f:
    last_line = line
    number_of_lines += 1
  cap = cv2.VideoCapture('../../AffWild2_test/VA_Set/videos/Test_Set/'+name[0])
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))+2
  print(total_frames)
  if(number_of_lines!=total_frames):
    for i in range (total_frames-number_of_lines):
      f.write(last_line)
  f.close()
