# TC-ResNet implementation with Keras

This is a reimplementation of the TC-ResNet 8 and 14 architecture, proposed by Hyperconnect. The research aims for a lightweight Convolutional Neural Network model to solve the Keyword Spotting problem with audio data in real time on mobile devices.

[Original paper](https://arxiv.org/abs/1904.03814v2)

[Author's implementation with Tensorflow](https://github.com/hyperconnect/TC-ResNet)

# How to use
Please download the dataset, and extract into a folder named `dataset` in the root folder of the repository.
The dataset augmentation is done using background noise files from Google Speech Command dataset

Run `main.py` to train the model.

Use the notebook TCResnet_KWS.ipynb incase facing difficulties running the file in local desktop.
Notebook link: https://colab.research.google.com/drive/1pQtTn3cRCArzAz8okPpB7ije17DhsNoS?usp=sharing

