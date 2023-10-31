# MNIST usando PyTorch

Welcome ! In this documentation, I will introduce the operation of a Neural Network designed for the classification of the well-known MNIST dataset(which will be expplained later).  
I'll use PyTorch as framework to build this.

# But what is MNIST?
MNIST is a widely recognized dataset in the fields of computer vision and machine learning. It comprises a collection of 70,000 grayscale images of handwritten digits from 0 to 9, each with a resolution of 28x28 pixels. The aim of this project is to train a classification model capable of correctly identifying the digits based on these images. You can check it out here: [Digit Recognizer | Kaggle](https://www.kaggle.com/competitions/digit-recognizer)

Without further ado. Let's dive into it.  

# Requirements and Imports
To perform MNIST classification, we're going to need some tools. I'll let it down below along with the imports.  

```python

```
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from customdataset import CustomMNISTDataset
import matplotlib.pyplot as plt

```
It's preferable to implement the imports as the code is being developed. But it's your choice.


