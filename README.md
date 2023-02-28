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

big_extract_features.py:  This file uses Spacy to extract features we consider to be useful for the role of SRL. We were able to implement them, but running the logistic regression with this train file did not work (it was too big). Therefore, we had to cut down on the number of features. The train-file of this can still be found in: https://drive.google.com/file/d/1Xmls18bIOBJ_qW8qkLlReP0bSg5f5xqE/view?usp=share_link, so that you don't need to extract this train file yourself. This file is extracted using Goolge Colab Pro, adding the train.csv file to the drive and mounting the drive. 
The input file is the csv file (train.csv, dev.csv and test.csv) from read_conllu_to_csv.py. The output is a csv file (separated by tab) with the extracted features (big_train.csv / big_dev.csv / big_test.csv).

new_extracted_features.py: This file uses Spacy to extract features we consider to be useful for the role of SRL and could be implemented. The input file is the csv file (train.csv, dev.csv and test.csv) from read_conllu_to_csv.py. The output is a csv file (separated by tab) with the extracted features (TRAIN_CSV.csv, DEV_CSV.csv / TEST_CSV.csv).

argument_identification.py:
used to get a binary gold-column which is used for argument identification. The values of the last column taking the output of 'new_extracted_features.py' : (TRAIN_CSV.csv and TEST_CSV.csv) are changed into zeros and ones. The output of this is then the same df, with a different 'Gold' column, see bin_TRAIN_CSV.csv, bin_TEST_CSV.csv and bin_DEV_CSV.csv (bin is short for binary). 

To get the performance of the argument identification, logistic regression is run using bin_TRAIN_CSV.csv and bin_TEST_CSV.csv using the logreg.py file. However, it is necessary to pay attention on how you split the files. If the file is separated by a tab, you change 
'components = line.rstrip('\n').split(',')' to 'components = line.rstrip('\n').split('\t')'. Else, you keep it as it is. 
 
logreg.py:
This file is based on the Machine Learning for NLP course, 2022. the same setup is used. For logreg, the hyperparameters have been tuned to not give a long runtime. 

big_logreg.py:
This file needs to be run when you want to check the output of the big_extract_features.py. The data is too big, that is why this big_logreg.py is not part of the results,  but is still included. 


All imports needed:
import conllu  
import pandas as pd  
import os  
import spacy  
import sys  
from sklearn.feature_extraction import DictVectorizer  
from sklearn.linear_model import LogisticRegression  
from sklearn import metrics  
from sklearn import classification_report
