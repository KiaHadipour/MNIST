
This repository contains a Python code that uses the MNIST dataset to train four different 
classification models to recognize handwritten digits.

## Installation

1. Clone the repository
2. Install the required packages using: pip install -r requirements.txt

## Usage

To train the model of your choice, run the following command: python KH-MNIST.py

## Results

![image](https://user-images.githubusercontent.com/33584311/228621313-c9695742-afe8-4ce0-9954-931edcb218fb.png)

![image](https://user-images.githubusercontent.com/33584311/228629178-8f5e199a-7451-408b-8c5f-f8805b9677ed.png)

Accuracy Score is a measure of how well a model predicts the correct output. Here it refers to the percentage of 
times that the model correctly predicted the label of each input. Training Time is the amount of time it takes 
for a model to process or "learn" from a dataset. Latency is the amount of time it takes for a model to make a 
prediction on a new input which is the time spent for predicting a single digit. Accordingly:

- The RFC had the highest accuracy score and the shortest training time. 
- The SVC had the second highest accuracy score and the longest training time.
- The KNC had the third highest accuracy score and the second shortest training time.
- The LR had the lowest accuracy score and the shortest latency.

## Disclaimer

Any reproduction or representation of the code published in this repository is contingent on referencing 
https://github.com/KiaHadipour/MNIST in your work.
