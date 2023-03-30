
This repository contains a Python code that uses the MNIST dataset to train four 
different classification models to recognize handwritten digits.

## Installation

Clone the repository and install the required packages via: pip install -r requirements.txt

## Usage

To train the model of your choice run the following command: python KH-MNIST.py

## Results

![image](https://user-images.githubusercontent.com/33584311/228629518-c07b972d-57b7-45cf-b4f5-e5eaa986ca01.png)

- The RFC had the highest accuracy score and the shortest training time.
- The SVC had the second highest accuracy score and the longest training time.
- The KNC had the third highest accuracy score and the second shortest training time.
- The LR had the lowest accuracy score and the shortest latency.

Accuracy Score: measures how well a model predicts the correct output. Here it refers to the percentage 
of times that the model correctly predicted the label of each input. 

Training Time: amount of time it takes for a model to process or "learn" from a dataset. 

Latency: amount of time it takes for a model to make a prediction on a new input or 
the time spent for predicting a single digit. 



