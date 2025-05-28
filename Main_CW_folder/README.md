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

To have a relatively accurate model; a basic model, that at least functions, had to be established first.

#### **Overall code:**

```python
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
from tensorflow.keras.datasets import cifar100
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Dropout, Activation
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

## https://www.geeksforgeeks.org/image-classification-using-cifar-10-and-cifar-100-dataset-in-tensorflow/
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

## https://github.com/LeoTungAnh/CNN-CIFAR-100/blob/main/CNN_models.ipynb
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255. 

classes = 100
ytrain_categories = to_categorical(y_train, num_classes=100)
yval_categories = to_categorical(y_val, num_classes=100)
ytest_categories = to_categorical(y_test, num_classes=100)

## https://github.com/uzairlol/CIFAR100-Image-Classification-CNN/blob/main/Item%20Image%20Model%20Training%20and%20Evaluation.ipynb
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

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, ytrain_categories, epochs=25, batch_size=64, validation_data=(x_val, yval_categories))

test_loss, test_accuracy = model.evaluate(x_test, ytest_categories)
print(f"Test accuracy: {test_accuracy * 100:.2f}%") 
```

#### ***1. Data Collection***
The CIFAR 100 dataset, that was planned on being used, was going to be from the tensor flow library.

#### ***2. EDA***

#### ***3. CNN model***

#### ***4. Prediction and Results***
Changing the amount of training data that was split into validation and training produced significant changes to accuracy. Compared to the original 90-10 split (training-validation), a split of 85-15 produced accuracies of 33.9, while a split of 95-5 produced 36.6%.

## <u><p align=center>**Second ML Pipeline Model**</u>

#### ***1. Data Collection***

#### ***2. EDA***

#### ***3. CNN model***

#### ***4. Prediction and Results***

## <u><p align=center>**Final ML Pipeline Model**</u>
This model improves and combines both the first and second models to produce higher accuracies in identifications.

#### ***1. Data Collection***

#### ***2. EDA***

#### ***3. CNN model***

#### ***4. Prediction and Results***



### <u><p align=center> **Future work**</u>

### <u><p align=center> **Libraries and Modules**</u>
#### Libraries:
- **Numpy:**
- **Matplotlib:**
- **Tensoreflow:**
- **Keras:**

#### Modules:
-


### <u><p align=center> **Issues and Bugs**</u>
Most issues occured during the construction of the code such as wrong labels used in certain functions. 

However, some bugs occur from the python kernel itself. In some cases the kernel would need a refresh before running the code so that results could appear.
### <u><p align=center> **Conclusions**</u>

##### <u><p align=center>  **References and Acknowledgments**</u>