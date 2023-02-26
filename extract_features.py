import spacy
import sys
import pandas as pd

def feature_extraction(inputfile, outputfile):
    
    '''
    Function that extracts potentially informative features from ConLL file for data exploration
    
    :param inputfile: path to ConLL file
    
    :type inputfile: string
    
    :return informative features: tsv file with informative features for each token in ConLL file
    '''

    
    conll_file = pd.read_csv(inputfile, delimiter='\t', header=None,  skipinitialspace = False, on_bad_lines='skip')
    df = pd.DataFrame(conll_file) 
    token_list = df[0].tolist()
    gold_list = df[2].tolist()
    token_string = ' '.join(map(str, token_list))
    
    
    

    nlp = spacy.load('en_core_web_sm')
    nlp.max_length = 5332026
    doc = nlp(token_string)


    data = []
    

    for tok in doc:
        token = tok.text 
        pos = tok.pos_
        lemma = tok.lemma_
        dependency = tok.dep_
        head = tok.head
        pos_head = head.pos_
        dependent = [t.text for t in tok.children]
        constituent = [t.text for t in tok.subtree]
        desc_dep = [t.dep_ for t in tok.subtree]
        desc_l =  [t.n_lefts for t in tok.subtree]
        desc_r =  [t.n_rights for t in tok.subtree]
        try:
            prev_tok = tok.nbor(-1)
        except IndexError:
            continue
        try:
            next_tok = tok.nbor()
        except IndexError:
            break
        prev_pos = prev_tok.pos_
        next_pos = next_tok.pos_
        morph = tok.morph
        iob = tok.ent_iob_
        ne = tok.ent_type_
        
        
 

        
            

          
       

        feature_dict = {'Token': token, 'PoS': pos, 'Lemma': lemma, 'Dependency': dependency, 'Head': head, 'Head POS': pos_head,
        'Dependent': dependent, 'Constituent': constituent, 'Previous POS': prev_pos, 'Next POS': next_pos,  'Morph': morph, 'IOB': iob, 
                        'NE': ne, 'Desc dep': desc_dep, 'Desc L': desc_l, 'Desc R': desc_r}
        data.append(feature_dict)

    
    df = pd.DataFrame(data=data)
    df['Gold'] = pd.Series(gold_list)                
    df.to_csv(outputfile,sep='\t', index = False) #change name of output file for dev data
    
    
    
def main(argv=None):
    
    if argv is None:
        argv = sys.argv
                
    inputfile = argv[1] #Path to ConLL train or dev file
    outputfile = argv[2]
    
    feature_extraction(inputfile)

    
    
if __name__ == '__main__':
    main()
    
