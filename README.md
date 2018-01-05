"# 02456 - project 11"

This repository contain the code used in 02456 - Deep learning. Project 11.

The zip files test_frame.zip and test.zip should be extracted before running the notebook.

The files single_layer_net.h5 and multiple_layer_net.h5 contain the trained models described in the paper. These are used to test the classification accuracy, generate confusion matrix and predict ball location on frames.

The notebook runs the prediction on the test set and predictions on a random frame, using both models. This code can also be found in the scripts prediction.py and test_frame_pred.py.

The script prediction.py predicts the classes of the data found in test.zip. The model to be used is specified at the top of the script. This generates two .txt files, one containing the predictions (0s and 1s), the other containing class probabilities. Further more a confusion matrix is generated and the total accuracy of the model is computed.

The script test_frame_pred.py predicts the classes of the data found in test_frame.zip (two different frames).

single_keras_net.py contain the training of the single layer ConvNet

keras_net.py contain the training of the multiple layer ConvNet
