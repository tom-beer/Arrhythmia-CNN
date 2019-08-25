# Arrhythmia-CNN
An implementation of the paper "Arrhythmia detection using deep convolutional neural network with long
duration ECG signals" (for the time being)

Use the script Create Arrhythmia Dataset.py to create the task dataset from the original MIT-BIH-Arrhythmia Dataset.
This function follows the description in the paper, and extracts arrhythmia 12 classes labeled both by number and by a string

Arrhythmia_Classifier.py containts the CNN according to the details in the paper. 

TODO:
- Split the dataset to train and test
- Write classification scores 
