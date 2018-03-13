import os

os.environ['OMP_NUM_THREADS'] = '4'

import torch
from torch.autograd import Variable
from torch import optim
# from torch.optim import lr_scheduler
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import math

train = pd.read_csv('../train.tsv', sep='\t')
test = pd.read_csv('../test.tsv', sep='\t')


def split_cat(text):
    try:
        return text.split("/")
    except:
        return "None", "None", "None"


train['general_cat'], train['subcat_1'], train['subcat_2'] = zip(*train['category_name'].apply(lambda x: split_cat(x)))
test['general_cat'], test['subcat_1'], test['subcat_2'] = zip(*test['category_name'].apply(lambda x: split_cat(x)))


def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0 / len(y))) ** 0.5


print("Handling missing values...")


def handle_missing(dataset):
    # dataset.category_name.fillna(value="na", inplace=True)
    dataset.brand_name.fillna(value="None", inplace=True)
    dataset.item_description.fillna(value="None", inplace=True)
    dataset.category_name.fillna(value="None", inplace=True)
    return (dataset)


train = handle_missing(train)
test = handle_missing(test)


# PROCESS CATEGORICAL DATA
# print("Handling categorical variables...")
def encode_text(column):
    le = LabelEncoder()
    le.fit(np.hstack([train[column], test[column]]))
    train[column + '_index'] = le.transform(train[column])
    test[column + '_index'] = le.transform(test[column])

