# For running experiments on our dataset (Auto-KWS)

Download train.zip from https://drive.google.com/drive/folders/12xsjNKHriCViExHLfMKJo4P55q36fSbY?usp=sharing

Extract the zip file to get a ( train ) folder in the same directory

# Additive Margin SincNet (AM-SincNet)
AM-SincNet is a new approach for speaker recognition problems which is based in the neural network architecture SincNet and the additive margin softmax  (AM-Softmax) loss function. It uses the architecture of the SincNet, but with an improved AM-Softmax layer.

This repository releases an example of code to perform a speaker recognition experiment on the TIMIT dataset. To run it with other datasets you can have a look at the instructions on the original SincNet repository (https://github.com/mravanelli/SincNet).

We should thank [@mravanelli](https://github.com/mravanelli/) for the [SincNet implementation](https://github.com/mravanelli/SincNet).

## Requirements
For running this experiment we used a Linux environment with Python 3.6.

You can see a list of python dependencies at [requirements.txt](requirements.txt).

To install it on conda virtual environment (`conda install --file requirements.txt`).

To install it on pip virtual environment (`pip install -r requirements.txt`).

## How to Run

Run the following from the same directory
``
python3 speaker_id.py --cfg=cfg/AKS_m065.cfg 
``

