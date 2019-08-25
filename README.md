# Arrhythmia-CNN
An implementation of the paper ["Arrhythmia detection using deep convolutional neural network with long
duration ECG signals"](https://www.sciencedirect.com/science/article/pii/S0010482518302713 "Link to paper")

Create Arrhythmia Dataset.py generates the task dataset from the original MIT-BIH-Arrhythmia Dataset.
This function follows the description in the paper, and extracts arrhythmia 12 classes labeled both by number and by a string

Arrhythmia_Classifier.py contains the CNN according to the details in the paper. 

TODO:
- Split the dataset to train and test
- Write classification scores 
