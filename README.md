# Ocular-Parking-System-v2

> Deep Learning Focused Parking Assistant System

## Requirements and dependencies

``` bash
# Python3
apt-get install python3.7-dev

# python3.7-pip
sudo apt-get install python3-pip
python3.7 -m pip install pip

# Pytorch
sudo -H python3.7 -m pip install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
# For GPU or other version: https://pytorch.org/get-started/locally/

# FastAI
python3.7 -m pip install --user fastai

# Pillow-5.1.0
python3.7 -m pip install --user -U Pillow

# OpenCV-4.1.1.26
python3.7 -m pip install --user opencv-python

# Jupyter
python3.7 -m pip install --user jupyterlab
sudo apt install jupyter-notebook
```

## Model

``` bash
# Download trained models
https://drive.google.com/open?id=1-6h-bckUGc3vtSDmbxg1UVV9Pm_ise7Y
```

## Inference

``` bash
# Example:
python3.7 main.py --video videoplayback2.mp4

# CLI Arguments:
* '--video' : Filename of input video located at img_input directory
* '--model_detection' : Filename of weights associated with detection
```

## Results

* Parkable Region Calibration

![Calibration](https://github.com/Skelliger7/Ocular-Parking-System-v2/tree/master/vid_output/1_calibrate.gif)

* Inference

![Inference](https://github.com/Skelliger7/Ocular-Parking-System-v2/tree/master/vid_output/2_inference.gif)

* Reservation

![Reservation](https://github.com/Skelliger7/Ocular-Parking-System-v2/tree/master/vid_output/3_reservation.gif)

* Overall

![overall](https://github.com/Skelliger7/Ocular-Parking-System-v2/tree/master/vid_output/4_overall.gif)
