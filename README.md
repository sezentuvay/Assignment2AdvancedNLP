# Assignment2AdvancedNLP
--
Run in the following order:
1) read_conllu_to_csv.py
2) extract_features.py
3) FOR ARGUMENT IDENTIFICATION:
    3.1) argument_identification.py
    3.2) logreg.py (change the names of the input and outpufiles in def main)
4) FOR ARGUMENT CLASSIFICATION:
    4.1) logreg.py (change the names of the input and outputfiles in the def main)

read_conllu_to_csv.py:
This file duplicates the sentences whenever there are multiple predicates, and gives those predicates' arguments to the next column. 
The input is a conllu file: (en_ewt-up-dev.conllu, en_ewt-up-test.conllu, en_ewt-up-train.conllu), and the output is: dev.csv, test.csv, train.csv.   

extract_features.py: This file uses Spacy to extract features we consider to be useful for the role of SRL. The input file is the output from read_conllu_to_csv.py 


argument_identification.py:
used to get a binary gold-column which is used for argument identification. The values of the last column taking the output of 'extract_features.py' : (train_new.csv / dev_new.csv / test_new.csv) are changed into zeros and ones. The output of this is then the same df, with a different 'Gold' column, see bin_train_new.csv, bin_test_new.csv and bin_dev_new.csv (bin is short for binary). 

To get the performance of the argument identification, logistic regression3 is run using bin_train_new.csv and bin_test_new.csv.

train_new.csv can be found in the Drive-link (https://drive.google.com/file/d/1Xmls18bIOBJ_qW8qkLlReP0bSg5f5xqE/view?usp=share_link), since this is a big file and takes a while to run. This file is 
extracted using Goolge Colab Pro, adding the train.csv file to the drive and mounting the drive. 


logreg.py:
This file is based on the Machine Learning for NLP course, 2022. the same setup is used. For logreg, the hyperparameters have been tuned to not give a long runtime. 

All imports needed:
import conllu  
import pandas as pd  
import os  
import spacy  
import sys  
from sklearn.feature_extraction import DictVectorizer  
from sklearn.linear_model import LogisticRegression  
from sklearn import metrics  
