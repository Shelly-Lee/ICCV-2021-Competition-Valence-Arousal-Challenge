# ICCV-2021-Competition-Valence-Arousal-Challenge

Submission to ICCV 2021: 2nd Workshop and Competition on Affective Behavior Analysis in-the-wild ([ABAW](https://ibug.doc.ic.ac.uk/resources/iccv-2021-2nd-abaw/))

Paper Link: https://arxiv.org/pdf/2107.03891.pdf

## Requirements:

 Resnet50 Pretrained Model ([resnet50_ferplus_dag](https://www.robots.ox.ac.uk/~albanie/pytorch-models.html))

 Aff-Wild2 Dataset : The competition organizers provide the cropped-aligned images.
## Installation
Use requirements.txt file
```bash
pip3 install -r requirements.txt
```
## How to run
Use CNN_feature_extraction.py file to save features
```bash
python3 CNN_feature_extraction.py --fps 30 --layer_name pool5_7x7_s1 --save_root Extracted_Features --data_root dir-to-aligned-face
```
Training (If you want to train arousal, then change valence to arousal in command)
```bash
python3 main.py --label_name valence --save_root valence
```
Testing
```bash
python3 main.py --label_name valence --save_root valence --test
python3 merge_test_predictions.py
python3 output.py
```
