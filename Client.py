## write a app that makes generated trained voices on singers on any mp3 file and upload it to uberduck.ai server
print("Writing request v0.1")
import requests
import json
import os
import sys
import time
import wave
import pyaudio
import numpy as np  
import wave 
import io
import subprocess       
import google as g
## write a api webhook to uberduck server
def write_request(file_name,file_path,file_type,file_size,file_duration,file_bitrate,file_channels,file_sampling_rate,file_sample_width,file_sample_width_bytes,file_sample_width_bits,file_sample_width_bytes_per_sample,file_sample_width_bits_per_sample,file_sample_width_bytes_per_sample_per_second,file_sample_width_bits_per_sample_per_second,file_sample_width_bytes_per_sample_per_second_per_channel,file_sample_width_bits_per_sample_per_second_per_channel,file_sample_width_bytes_per_sample_per_second_per_channel_per_frame,file_sample_width_bits_per_sample_per_second_per_channel_per_frame,file_sample_width_bytes_per_sample_per_second_per_channel_per_frame_per_sample,file_sample_width_bits_per_sample_per_second_per_channel_per_frame_per_sample,file_sample_width_bytes_per_sample_per_second_per_channel_per_frame_per_sample_per_channel,file_sample_width_bits_per_sample_per_second_per_channel_per_frame_per_sample_per_channel,file_sample_width_bytes_per_sample_per_second_per_channel_per_frame_per_sample_per_channel_per_frame,file_sample_width_bits_per_sample_per_second_per_channel_per_frame_per_sample_per_channel_per_frame_per_sample,file_sample_width_bytes_per_sample_per_second_per_channel_per_frame_per_sample_per_channel_per_frame_per_sample_per_channel,file_sample_width_bits_per_sample_per_second_per_channel_per_frame_per_sample_per_channel_per_frame_per_sample_per_channel_per_frame,file_sample)  
    ## write a api webhook to uberduck server
    print("Writing request v0.1")
    print("file_name:",file_name)
    print("file_path:",file_path)
    print("file_type:",file_type)
    print("file_size:",file_size)
    print("file_duration:",file_duration)
    print("file_bitrate:",file_bitrate)
    ### ask the use what uberduck api key is being used
    api_key = input("What is the api key being used? ")
    print("api_key:",api_key)
    ### ask the use what uberduck api url is being used
    ## connect to uberduck.ai server
    uberduck_url = "https://api.uberduck.ai/v1/requests"
    ## ask what your username and passwrord and email is
    username = input("What is your username? ")
    print("username:",username)
    password = input("What is your password? ")
    print("password:",password)
    email = input("What is your email? ")
    print("email:",email)
    ### ask the use what uberduck api url is being used
    ## connect to uberduck.ai server
    uberduck_url = "https://api.uberduck.ai/v1/requests"
    ## ask where to get the mp3 file from your folder in a dataset and save it as as a .json file
    file_path = input("What is the path to the mp3 dataset? ")
    print("file_path:",file_path)
    ### ask the use what uberduck api url is being used
    ## connect to uberduck.ai server
    uberduck_url = "https://api.uberduck.ai/v1/requests"
    ## request the uberduck to train the model
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
## build a dataset with tensorflow models
def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    import collections  # for OrderedDict
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
## save the file as a uberduck compatible datasets
def save_file(file_name, file_path, file_type, file_size, file_duration, file_bitrate):
    import json
    import requests
    import os
    import time
    import datetime
    import sys
    import csv
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_recall_curve
    ## connect to uberduck.ai server
    uberduck_url = "https://api.uberduck.ai/v1/requests"
    ## ask what your username and passwrord and email is
    username = input("What is your username? ")
    print("username:",username)
    password = input("What is your password? ")
    print("password:",password)
    email = input("What is your email? ")
    print("email:",email)
    ### ask the use what uberduck api url is being used
    ## connect to uberduck.ai server
    uberduck_url = "https://api.uberduck.ai/v1/requests"
    ## ask where to get the mp3 file from your folder in a dataset and save it as as a .json file
    file_path = input("").strip()
    print("file_path:",file_path)
    ### ask the use what uberduck api url is being used
    ## connect to uberduck.ai server
    uberduck_url = "https://api.uberduck.ai/v1/requests"
    ## done
    ## connect to uberduck.ai server
    print(################################)
    print("training complete.")
def upload_data(file_path, file_type, file_size, file_duration, file_bitrate):
    