from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report


### Based on the Machine Learning for NLP course, 2022. 

def extract_features_and_labels(trainingfile):

    '''
    Function that extracts features and gold labels 
    
    :param trainingfile: path to ConLL file
    
    :type trainingfile: string
    
    :return features: list containing dictionary mapping tokens to features
    :return gold: list of gold labels for each token
    '''
    features = []
    gold = []
    with open(trainingfile, 'r', encoding='utf-8') as infile:
        for line in infile:
            components = line.rstrip('\n').split('\t')
            if len(components) > 0:
                token = components[0]
                pos = components[1]
                lemma = components[2]
                dependency = components[3]
                head = components[4]
                dependent = components[5]
                constituent = components[6]
                gold_a = components[-1]
                feature_dict = {'Token': token, 'PoS': pos, 'Lemma': lemma, 'dependency': dependency, 'head': head, 'dependent': dependent, 'constituent': constituent}
                features.append(feature_dict)
                gold.append(gold_a)
    return features, gold

def extract_features(inputfile):
    
    '''
    Function that extracts features  
    
    :param inputfile: path to TSV file
    
    :type inputfile: string
    
    :return inputdata: list containing dictionary mapping tokens to features
    '''
   
    inputdata = []
    with open(inputfile, 'r', encoding='utf-8') as infile:
        for line in infile:
            components = line.rstrip('\n').split('\t')
            if len(components) > 0:
                token = components[0]
                pos = components[1]
                lemma = components[2]
                dependency = components[3]
                head = components[4]
                dependent = components[5]
                constituent = components[6]
                gold_a = components[-1]
                feature_dict = {'Token': token, 'PoS': pos, 'Lemma': lemma, 'dependency': dependency, 'head': head, 'dependent': dependent, 'constituent': constituent}
                inputdata.append(feature_dict)
    return inputdata

def extract_gold_labels_input(inputfile):
    ''' Function that gives the gold labels of the inputfile, meaning the testfile

    :param inputfile: path to the inputfile
    :type inputfile: string
    :return gold_labels: list of the gold labels (last colum)
    
    '''
    gold_labels = []
    with open(inputfile, 'r', encoding='utf-8') as infile:
        for line in infile:
            components = line.rstrip('\n').split('\t')
            if len(components) > 0:
                gold_a = components[-1]
                gold_labels.append(gold_a)
    return gold_labels

def create_classifier(features, gold):
    ''' Function that takes feature-value pairs and gold labels as input and trains a Logistic Regression model
   
    :param features: feature-value pairs
    :param gold: gold labels
    :type features: a list of dictionaries
    :type gold: a list of strings
    
    :return classifier: a trained Logistic Regression classifier
    :return vec: a DictVectorizer to which the feature values are fitted. '''
  
    model = LogisticRegression(solver='saga', max_iter=1000)
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(features)
    model.fit(features_vectorized, gold)

    return model, vec

def classify_data(model, vec, inputdata):
    
    '''
    Function that performs classification of samples and outputs file mapping predicted labels to gold labels
    
    :param model: trained model
    :param vec: trained DictVectorizer
    :param inputdata: input file to be classified
  
    :type inputdata: string
    '''
    features = extract_features(inputdata)
    features = vec.transform(features)
    predictions = model.predict(features)
    return predictions


def print_precision_recall_fscore(predictions, gold):
    '''
    Function that prints out precision, recall and f-score
    
    :param predictions: predicted output by classifier
    :param goldlabels: original gold labels
    :type predictions, goldlabels: list of strings
    '''
    report = classification_report(gold, predictions)
    print(report, sep='\t')


def main(argv=None):
          
    trainingfile = 'data/output/TRAIN_CSV.csv' #Training file
    inputfile = 'data/output/TEST_CSV.csv' #Development (DEV_CSV.csv) or Test file 

    test_gold_labels = extract_gold_labels_input(inputfile)
    training_features, gold_labels = extract_features_and_labels(trainingfile)
    ml_model, vec = create_classifier(training_features, gold_labels)
    predictions = classify_data(ml_model, vec, inputfile)
    print_precision_recall_fscore(predictions, test_gold_labels)
    
if __name__ == '__main__':
    main()


