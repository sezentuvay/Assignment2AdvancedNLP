from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import sys

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
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0]
                pos = components[1]
                lemma = components[2]
                dependency = components[3]
                pos_head = components[4]
                dependent = components[5]
                constituent = components[6]
                previous_pos = components[7]
                next_pos = components[8]
                morph = components[9]
                iob = components[10]
                ne = components[11]
                desc_dep = components[12]
                desc_l = components[13]
                desc_r = components[14]
                gold_a = components[-1]
                    
                 feature_dict = {'Token': token, 'PoS': pos, 'Lemma': lemma, 'Dependency': dependency, 'Head': head, 'Head POS': pos_head,
        'Dependent': dependent, 'Constituent': constituent, 'Previous POS': prev_pos, 'Next POS': next_pos,  'Morph': morph, 'IOB': iob, 
                        'NE': ne, 'Desc dep': desc_dep, 'Desc L': desc_l, 'Desc R': desc_r}
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
            components = line.rstrip('\n').split()
            if len(components) > 0:
                token = components[0]
                pos = components[1]
                lemma = components[2]
                dependency = components[3]
                pos_head = components[4]
                dependent = components[5]
                constituent = components[6]
                previous_pos = components[7]
                next_pos = components[8]
                morph = components[9]
                iob = components[10]
                ne = components[11]
                desc_dep = components[12]
                desc_l = components[13]
                desc_r = components[14]
                gold_a = components[-1]
                feature_dict = {'Token': token, 'PoS': pos, 'Lemma': lemma, 'Dependency': dependency, 'Head': head, 'Head POS': pos_head,
        'Dependent': dependent, 'Constituent': constituent, 'Previous POS': prev_pos, 'Next POS': next_pos,  'Morph': morph, 'IOB': iob, 
                        'NE': ne, 'Desc dep': desc_dep, 'Desc L': desc_l, 'Desc R': desc_r}
                features.append(feature_dict)
                gold.append(gold_a)
    return inputdata



def create_classifier(features, gold):
    ''' Function that takes feature-value pairs and gold labels as input and trains a Logistic Regression model
   
    :param features: feature-value pairs
    :param gold: gold labels
    :type features: a list of dictionaries
    :type gold: a list of strings
    
    :return classifier: a trained Logistic Regression classifier
    :return vec: a DictVectorizer to which the feature values are fitted. '''
  
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(features)
    model.fit(features_vectorized, gold)

    return model, vec

def classify_data(model, vec, inputdata, outputfile):
    
    '''
    Function that performs classification of samples and outputs file mapping predicted labels to gold labels
    
    :param model: trained model
    :param vec: trained DictVectorizer
    :param inputdata: input file to be classified
    :param outputfile: new file containing gold and predicted labels
  
    :type inputdata: string
    :type outputfile: string
    
    :return ouputfile: ConLL file mapping predicted labels to gold labels
    
    '''
    features = extract_features(inputdata)
    features = vec.transform(features)
    predictions = model.predict(features)
    
    return predictions
    
def print_confusion_matrix(predictions, gold):
    '''
    Function that prints out a confusion matrix
    
    :param predictions: predicted labels
    :param goldlabels: gold standard labels
    :type predictions, goldlabels: list of strings
    '''   
    
    #based on example from https://datatofish.com/confusion-matrix-python/ 
    data = {'Gold':    gold, 'Predicted': predictions    }
    df = pd.DataFrame(data, columns=['Gold','Predicted'])

    confusion_matrix = pd.crosstab(df['Gold'], df['Predicted'], rownames=['Gold'], colnames=['Predicted'])
    print (confusion_matrix)


def print_precision_recall_fscore(predictions, gold):
    '''
    Function that prints out precision, recall and f-score
    
    :param predictions: predicted output by classifier
    :param goldlabels: original gold labels
    :type predictions, goldlabels: list of strings
    '''
    
    precision = metrics.precision_score(y_true=gold,
                        y_pred=predictions,
                        average='macro')

    recall = metrics.recall_score(y_true=gold,
                     y_pred=predictions,
                     average='macro')


    fscore = metrics.f1_score(y_true=gold,
                 y_pred=predictions,
                 average='macro')

    print('P:', precision, 'R:', recall, 'F1:', fscore)
    
    
    
def main(argv=None):
    
    if argv is None:
        argv = sys.argv
                
    trainingfile = argv[1] #Training file
    inputfile = argv[2] #Development or Test file
    modelname = argv[3] #'logreg' 

    
    feature_values, labels = extract_features_and_gold_labels(trainfile, selected_features=selected_feature)
    model, vectorizer = create_vectorizer_and_classifier(feature_values, labels, modelname)
    predictions, goldlabels = get_predicted_and_gold_labels(testfile, vectorizer, model, selected_feature)
    print_confusion_matrix(predictions, goldlabels)
    print_precision_recall_fscore(predictions, goldlabels)
    
    
if __name__ == '__main__':
    main()
    
