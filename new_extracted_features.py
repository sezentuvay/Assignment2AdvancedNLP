import sys
import pandas as pd
import spacy
from spacy.tokens import Doc
import csv


def extract_features(inputfile, outputfile):
    
    '''
    Function that extracts features from ConLL file to feed into Logistic Regression model
    
    :param input_path: path to preprocessed ConLL file
    :param output_path: path to TSV file with extracted features
    
    
    :type input_path: string
    :type output_path: string
    
    :return extracted features: tsv file with features for each token in ConLL file
    '''
    

    infile = pd.read_csv(inputfile, delimiter=',', header=None,  skipinitialspace = False, on_bad_lines='skip')
    df = pd.DataFrame(infile) 
    token_list = df[2].tolist()
    gold_list = df[12].tolist()

    t_list = ''.join(token_list)
    
    #create custom tokenizer

    data = []
    processed_tokens = []
    for token in token_list:
        newtoken = str(token)
        processed_tokens.append(newtoken)
    joined_tokens = " ".join(processed_tokens)

    def custom_tokenizer(text):
        tokens = df[2].tolist()
        return Doc(nlp.vocab, tokens)
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = custom_tokenizer
    nlp.max_length = 5169840 #adjust max length if needed
    text = joined_tokens

    doc = nlp(text)

    #create features to extract

    for tok in doc:
        token = tok.text
        pos = tok.pos_
        lemma = tok.lemma_
        dependency = tok.dep_
        head = tok.head
        dependent = [t.text for t in tok.children]
        constituent = [t.text for t in tok.subtree]
        

        
        feature_dict = {'Token': token, 'PoS': pos, 'Lemma': lemma, 'dependency': dependency, 'head': head,
                        'dependent': dependent,
                        'constituent': constituent}
        data.append(feature_dict)


    # save results to tsv file
    df = pd.DataFrame(data=data)
    df['Gold'] = gold_list  # append gold labels at the end
    df.to_csv(outputfile, sep='\t', header=True, index=False)



      
    
    
def main():
    inputfiles = ['test', 'dev', 'train']
    for inputfile in inputfiles:
        extract_features('data/output/'+inputfile+'.csv', 'data/output/'+inputfile.upper()+'_CSV.csv')

main()
    
