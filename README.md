## LSTM Model for NLP
 The process involves cleaning and preparing a dataset of news articles by tokenizing and encoding the text and labels, splitting the data into training and testing sets, and building and training a LSTM model. The model's performance is evaluated using a classification report and confusion matrix, and the model and associated preprocessing objects are saved to files.

The script also includes several helper functions and modules that are called throughout the pipeline, including a custom text_cleaning function for cleaning the text data and a lstm_model_creation function for creating the LSTM model. These functions and modules are intended to improve the readability and modularity of the code.

## Badges

![Windows 11](https://img.shields.io/badge/Windows%2011-%230079d5.svg?style=for-the-badge&logo=Windows%2011&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

## Acknowledgement
I would like to acknowledge the following sources that were used in the development of this code:

The dataset of news articles from the BBC, which was obtained from the following URL: https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv
The text_cleaning function, which was inspired by the tutorial "Natural Language Processing with Python" by Susan Li at PyCon Canada 2019: https://github.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial

The lstm_model_creation function, which was adapted from the tutorial "Sequence Classification with LSTM Recurrent Neural Networks in Python with Keras" by Jason Brownlee: https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

The TensorFlow and scikit-learn libraries, which were used extensively throughout the script for tasks such as tokenization, padding, one-hot encoding, model creation and training, and evaluation.
