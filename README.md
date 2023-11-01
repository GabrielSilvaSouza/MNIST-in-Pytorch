# MNIST using PyTorch

<p align="center">
    <img src="figure1.webp" alt="My image">
</p>

Welcome ! In this documentation, I will introduce the operation of a Neural Network designed for the classification of the well-known MNIST dataset(which will be expplained later).  
I'll use PyTorch as framework to build this.

# But what is MNIST?
MNIST is a widely recognized dataset in the fields of computer vision and machine learning. It comprises a collection of 70,000 grayscale images of handwritten digits from 0 to 9, each with a resolution of 28x28 pixels. The aim of this project is to train a classification model capable of correctly identifying the digits based on these images. You can check it out here: [Digit Recognizer | Kaggle](https://www.kaggle.com/competitions/digit-recognizer)

Without further ado. Let's dive into it.  

# Requirements and Imports
To perform MNIST classification, we're going to need some tools. I'll let it down below along with the imports.  

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

# Knowing our Dataset  

Let's analyse some aspects of our dataset. First of all, it is composed by 784 columns - representing the 28x28 = 784 pixels - and one label - which is the given number for a certain pattern of pixels.  
```python
dt = pd.read_csv('mnist_dataset.csv')
dt.iloc[1,1:]

'''
Output:

pixel0      0
pixel1      0
pixel2      0
pixel3      0
pixel4      0
           ..
pixel779    0
pixel780    0
pixel781    0
pixel782    0
pixel783    0
Name: 1, Length: 784, dtype: int64
'''
```

We also have 42000 rows which will be used to train and test our NN model. Later, we will divide this dataset into two parts in the proportion of 80% for the training set and the rest for the testing set.  

```python
dt.shape[0]

'''
Output:
42000
'''
```

That's enough for now ! Of course we could get deeper into the dataset details, but it would be against and beyond the purpose of this "tutorial". Now, let's build the custom dataset.

# Customising dataset

Here is where things start to get interesting and we start using real code ! And more, starts tasting PyTorch.  
In this step, we are going to build a class in which will perform the preparation of our data to feed the NN.  

1. Create a .py file called `customdataset`, or whatever name you like. But remember that you'll use that name later in another file.
2. Import some useful frameworks:
   ```python
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import Dataset
   ```
3. Create a class (called `customMNISTDataset` or, again, whatever name you like) and implement the methods `def __init__()`, `def __len__()`, `def __getitem__()`. These methods are essential(*exactly as written*) for our purposes and enable our custom dataset to seamlessly integrate with PyTorch's data loading utilities like DataLoader.
   1. This class will inherit some aspects from the PyTorch `Dataset` class in order to facilitate data handling and transformations.
   2. `__init__()`: This constructor is used to initialize the dataset and prepare it for use.
   ```python
       def __init__(self, csv_file, transform=None):
        data = pd.read_csv(csv_file)
        self.labels = data['label'].values.astype(int)
        self.images = data.drop('label', axis=1).values.astype(float)
        self.transform = transform
   ```
   In the second line, we use pandas(pd) to load the data and perform preprocessing. After that, the code extract the labels of each pixel with `self.labels = data['label'].values.astype(int)`, returning a 1-D array. In `self.images = data.drop('label', axis=1).values.astype(float)`, the pixels columns are taken by droping the column label - notice that since the label are destined to numbers classification between 0-9 (in the sets of natural numbers), it must be *integet(int)*.
   3. `__len__()`: This constructor just returns the lenght of the dataset. It is useful though, because our Dataloader(which will be explained later) will use it.
    ```python
    def __len__(self):
        return len(self.labels)
   ```
    4. `__getitem__()`: This method is responsible for retrieving a specific sample from your dataset given an index. Besides, it allows us to apply some useful operations such as data conversion, reshaping and transforms. It returns a dict or tuple with the label and its respective pixels(image).
   ```python
    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28).astype(np.float32)  
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
   ```

   
   




