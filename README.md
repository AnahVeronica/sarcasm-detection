# sarcasm-detection
### Softwares and libraries
**1) Pandas**
**2) Numpy**
**3) Sklearn:**
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

**4) TextBlob**
from textblob import TextBlob

**5) SciPy**
from scipy.stats import entropy

**6) Spacy Text Processing**
import spacy
from spacy.tokenizer import Tokenizer
import en_core_web_sm
import en_core_web_md
**7)Visualizations**
Plotly
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

**8) Keras**
import keras
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout,Bidirectional

**9) Tensorflow**
import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants

**10) Sagemaker api**
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.predictor import csv_serializer

**11) Additional required file**
import boto3, re
import io
import json
import shutil
import tarfile
from keras.preprocessing import text, sequence