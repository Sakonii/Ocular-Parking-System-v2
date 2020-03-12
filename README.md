# Ocular-Parking-System-v2

Deep Learning Focused Parking Assistant System

<div align="center">
  <img src="https://github.com/Skelliger7/Ocular-Parking-System-v2/blob/detectron-implementation/vid_output/out.png" width="596" height="292">
</div>

## Requirements and dependencies

Python3.7 and pip

```bash

# Python3
apt-get install python3.7-dev

# python3.7-pip
sudo apt-get install python3-pip
python3.7 -m pip install pip
```

Dependencies installation under pip package manager

``` bash

# Pytorch
sudo -H python3.7 -m pip install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
# For GPU or other version: https://pytorch.org/get-started/locally/

# OpenCV-4.1.1.26
python3.7 -m pip install --user opencv-python

# folium-0.10.1
python3.7 -m pip install --user folium

# Jupyter
python3.7 -m pip install --user jupyterlab
sudo apt install jupyter-notebook
```

Build detectron2 from source:

``` bash
# Detectron2-0.1.1
https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md

# update 'detectron2/model_zoo/config' as in:
https://drive.google.com/drive/folders/14rL_GI_EleuqtyBow_AfudSd7llisV9u?usp=sharing
```

## Inference

``` bash
# Example:
python3.7 main.py --video videoplayback2.mp4

# CLI Arguments:
* '--video' : Filename of input video located at img_input directory
* '--model_detection' : Filename of weights associated with detection
* '--cfg_path' : Path to model cfg file relative to 'detectron2/model_zoo'
```

## Results

* Parkable Region Calibration

<div align="center"><img src="https://github.com/Skelliger7/Ocular-Parking-System-v2/blob/detectron-implementation/vid_output/1_calibrate.gif" width="320" height="289"></div>

* Inference

<div align="center"><img src="https://github.com/Skelliger7/Ocular-Parking-System-v2/blob/detectron-implementation/vid_output/2_inference.gif?raw=true" width="320" height="289"></div>

* Overall

<div align="center"><img src="https://github.com/Skelliger7/Ocular-Parking-System-v2/blob/detectron-implementation/vid_output/4_overall.gif?raw=true" width="320" height="321"></div>
