# R3DCNN
this project implements the hand gesture recognition algorithm introduced in paper [online detection and classification of dynamic hand gestures with recurrent 3d convolutional neural networks](http://research.nvidia.com/sites/default/files/publications/NVIDIA_R3DCNN_cvpr2016.pdf)

### introduction
this project is an implement of dynamic hand gesture recognition algorithm which takes advantages of both C3D and LSTM. because part of the model is adopted from C3D, the model is finetuned from trained models from C3D which is provided from [another repo](https://github.com/breadbread1984/c3d) of my git.

### how to create dataset
download the dataset [here](https://drive.google.com/a/nvidia.com/folderview?id=0ByhYoRYACz9cMUk0QkRRMHM3enc&usp=sharing) and unzip it under the current directory. then use the following command to generate dataset.

```bash
python create_dataset.py
```

### how to train
place the trained C3D model directory under current directory and train with command

```bash
python train_r3dcnn.py
```

### how to test
test on video captured from webcam with command

```bash
python ActionRecognition.py
```
