"# 02456 - project 11"

This repository contain the code used in 02456 - Deep learning. Project 11.

The files single_layer_net.h5 and multiple_layer_net.h5 contain the trained models described in the paper. These are used to test the classification accuracy, generate confusion matrix and predict ball location on frames.

The script prediction.py predicts the classes of the data found in test.zip. The model to be used is specified at the top of the script.

The script test_frame_pred.py predicts the classes of the data found in test_frame.zip (two different frames).

single_keras_net.py contain the training of the single layer ConvNet
keras_net.py contain the training of the multiple layer ConvNet
