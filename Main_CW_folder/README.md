# <p align=center>Big Data: Main Coursework

#### <p align=center>Student Numbers: up2014075, up2161618

### <u><p align=center>**Introduction** </u>

 CIFAR-100 is a dataset that consists of over 60,000 images that can be distributed across 100 different classifications. This dataset is recognized for its use in benchmarking image classifications models. The model created throughout this course work is a Convolutional Neural Network (CNN) model. This report will describe the basic CNN model architecture neeed as well as the convolution layers, pooling strategies and fully connected layers to identify images. The CNN model will then undergo improvements to increase the accuracy in identificating images. 

### <u><p align=center>**Business Objectives** </u>
The purpose of the project is to create and evaluate a CNN model, that is capable of recognizing images across different categories within the CIFAR 100 dataset by:

- ##### Achieving highly accurate results when classifying images into the 100 distinct classes
- ##### Ensuring model functions and produces results when tested with unseen data 
- ##### Improving and evaluating the CNN architecure to improve efficiency without compromising accuracy

## <u><p align=center>**First ML Pipeline Model**</u>

To understand how the model works; a basic model, that at least functions, had to be established first.

#### **Overall code:**

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
from tensorflow.keras.datasets import cifar100
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Dropout, Activation
from sklearn.model_selection import train_test_split

#Load data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

#Checking data and array shape 
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#Splitting training data into training and validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

#https://www.geeksforgeeks.org/image-classification-using-cifar-10-and-cifar-100-dataset-in-tensorflow/
def show_samples(data, labels):
    plt.subplots(figsize=(10, 10))
    for i in range(12):
        plt.subplot(3, 4, i+1)
        k = np.random.randint(0, data.shape[0])
        plt.title(int(labels[k]))
        plt.imshow(data[k])
    plt.tight_layout()
    plt.show()

show_samples(x_train, y_train)


#Processing data 
#Converting pixels to float type
#https://github.com/LeoTungAnh/CNN-CIFAR-100/blob/main/CNN_models.ipynb
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

#One hot encoding to target classes 
classes = 100
ytrain_categories = to_categorical(y_train, num_classes=100)
yval_categories = to_categorical(y_val, num_classes=100)
ytest_categories = to_categorical(y_test, num_classes=100)

#Building CNN model 
#Uses layers as a 'filtering' system that making model learn based on patterns from training
#https://github.com/uzairlol/CIFAR100-Image-Classification-CNN/blob/main/Item%20Image%20Model%20Training%20and%20Evaluation.ipynb
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(100, activation='softmax')
])

model.summary()


#Beginning the training of model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Main training section
#chatgpt helped to structure how model can undergo training
history = model.fit(x_train, ytrain_categories, epochs=25, batch_size=64, validation_data=(x_val, yval_categories))

test_loss, test_accuracy = model.evaluate(x_test, ytest_categories)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

#Visualisation of results through graphs using matplotlib
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
```

#### ***1. Data Collection and Processing***
The CIFAR 100 dataset, that was planned on being used, was going to be from the tensor flow (Keras) library using the ``` cifar100.load_data() ``` from ```tensorflow.keras.datasets```. Tensor flow was a much smoother option compared to the process of unpickling, when retrieving it from another source.

#### ***2. Exploratory Data Analysis (EDA)***
``` python
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

show_samples(x_train, y_train)
```
This section prints and displays the shape of the training and test datasets to confirm dimension. It also shows a random sample image from the dataset using the Matplotlib. 

#### ***3. CNN model***
To filter the data and patterns from the images, a layered architecture was implemented:

- Convolutional layering (`Conv2D`) was first used to extract features from the images by applying filters.
- Pooling Layers (`Maxpooling2D`) was then used to reduce spatial dimensions in feature maps.
- Flattening layers (`Flatten`) transformed 2D feature maps into 1D vectors.
- `Dense` Layers were then applied to connect the layers to be classified.
- `Dropout` layer prevented overfitting by randomly setting nodes to zer during the training phase.

To ensure the model was ready for training it had to be compiled using:
```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
- Optimizer (`Adam`) is an adaptive optimizer that dynamically adjusts parameter updates based momentary estimation, convergence speed and model performance.
- The `loss` function, with `categorical_crossentropy`, quantifies the difference between predicted distributions and the actual labels.
- The `Metric` measures the fraction of correct predictions providing a performance metric for the model's overall effectiveness.

#### ***4. Training Model***
```python
history = model.fit(x_train, ytrain_categories, epochs=25, batch_size=64, validation_data=(x_val, yval_categories))
```
This training is conducted over 25 epochs using batch processing, as a method for more effeicient computation.

This gives out data that is from monitoring the training and validation accuracy and loss per epoch.

#### ***5. Prediction, Results and Evaluation***
```python
test_loss, test_accuracy = model.evaluate(x_test, ytest_categories)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
```
The model was then finally tested and evaluated with unseen test data.

Base accuracy result, after running the model, was 36.9%.

Changing the amount of training data that was split into validation and training produced significant changes to accuracy. Compared to the original 90-10 split (training-validation), a split of 85-15 produced accuracies of 33.9, while a split of 95-5 produced 36.6%.

Increasing width and range of the convolution and other related layers resulted in higher accuracies of around 39%:
```python 
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(100, activation='softmax')
]) 
```
was turned to
``` python 
model = keras.models.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(100, activation='softmax')
])
```
Increasing the Epoch to 50 and 100 also did result in an increase in accuracy, by 1-2%, as it did allow more runs through the training data. This in turn allowed the accuracy of the model to plateau and stabilise at its maximum. However, this method only allowed the performance to reach its maximum rather than actually amplifying the performance.
## <u><p align=center>**Second ML Pipeline Model**</u>

#### **Overall Code:**
``` python

```
#### ***1. Data Collection and Processing***

#### ***2. EDA***

#### ***3. CNN model***

#### ***4. Training***

#### ***5. Prediction, Results and Evaluation***

## <u><p align=center>**Third ML Pipeline Model**</u>
This model improves and combines both the first and second models to produce higher accuracies in identifications.

#### **Overall Code:**
``` python

```

#### ***1. Data Collection and Processing***

#### ***2. EDA***

#### ***3. CNN model***

#### ***4. Training***

#### ***5. Prediction, Results and Evaluation***



### <u><p align=center> **Future work**</u>
With a working model the primary objectives for future work would be to improve the processing time taken, when training the model, for each epoch as well as increasing the accuracy.

Processing time could be cut by introducing batch normalization, which normalizes layer input making training faster and more stable. 

On the other hand, to increase accuracy, image augmentation can be used to keep the same CNN architecture and prevent the need of other model architectures. Through augmentation the training can be enhanced, while also reducing overfitting.

### <u><p align=center> **Libraries and Modules**</u>
#### Libraries:
- **NumPy:**
NumPy is a python library used for numerical operations and especially for array management. 
- **Matplotlib:**
This library is mostly used for visualization and plotting data nad results. 
- **Tensorflow:**
The main library used for machine learning within this code. It's usually used for deep leaning and neural networks. This is used within the codes to provide the CIFAR 100 dataset (through Keras) as well as the network creation, compilation and training. 
- **Keras:**
This library is built ontop of Tensorflow and is used for defining and training neural netweorks. Within the code it has been used to provide the CIFAR 100 dataset, components to the CNN model, such as layers, and utilities like ``` to_categorical``` for one hot encoding.
- **sklearn:**
Sklearn library provides the tools to split data (e.g. training and validation data) as well to evaluate models.

#### Modules:
- **keras.datasets:**
This module is used to access popular datasets like CIFAR 10 and CIFAR 100.
- **Keras.utils:**
Provides utilities for handling and preprocessing data like one-hot encoding.
- **Keras.layers:**
Module for defining various neural network layers, like a filtering system.
- **Keras.models:**
This allows neural network models to be defined, compiled and managed.
- **tensorflow.keras.datasets:**
same as `keras.datasets` as it provides specific dataset loaders.

### <u><p align=center> **Unresolved Issues and Bugs**</u>
Most issues occured during the construction of the code such as wrong labels used in certain functions. However, some issues did occur from the python kernel itself. In some cases the kernel would crash and required a refresh before running the code so that it could function again. Possible causes could be because of internet connection issues or amount of space of the cpu.
### <u><p align=center> **Conclusions**</u>
Over this project, a model has successfully been created that predicts and identifies, with a 39% accuracy, images from the CIFAR 100 dataset into a certain category.

The different overall codes and CNN models have also given a basic understanding of data processing for ML pipelines and CNN architecture, when used for real world applications such as image identificators.

### <u><p align=center>  **References and Acknowledgments**</u>
(https://www.geeksforgeeks.org/image-classification-using-cifar-10-and-cifar-100-dataset-in-tensorflow/) This source helped to structure and show the EDA for the frist model

(https://github.com/LeoTungAnh/CNN-CIFAR-100/blob/main/CNN_models.ipynb) This example of an existing CIFAR 100 was used to understand data processing and implementing one-hot encoding during the creation of the first model.

(https://github.com/uzairlol/CIFAR100-Image-Classification-CNN/blob/main/Item%20Image%20Model%20Training%20and%20Evaluation.ipynb) Another example was used, for model one, to produce the CNN architecture through the different layering modules.


### <u><p align=center>  **Terms and Definitions**</u>
- *Overfitting:* Situation where model learns training too well, including the noise and random fluctuations, that then results in poor generalization of new unseen data. 
- *Optimizer:* An algorithm employed during training phase to minimize loss function by iterative updates to model parameters. 
- *Epoch:* Means a complete pass through of the entire training dataset during training phase. 
- *One-Hot Encoding:* A representation method that converts categorical variables into binary vectors. In this case it turns data from the categorical labels to binary vectors.
- *Categorical Cross-Entropy:* `Loss` function specifically designed for multi-class classification problems, that measures discrepancy between predicted probability distributions and true distributions that are represented through as one hot encoded vector.
- *Convergence:* Referring to the condition where optimization alogrithm reaches a stable state. This seen with characteristics such as minimal changes in loss values between each epoch. This means model has adequetly learned the patterns.
- *Augmnetation:* Is a method that adds more data by modifying current data through transformations. This increases the amount of data for training.
- *Normalization:* This is when data is converted to a more standard/uniform format so that it can be easier to work with and analyze.