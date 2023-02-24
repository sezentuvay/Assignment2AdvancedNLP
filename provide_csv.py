import conllu
import pandas as pd
import os

def read_conllu_write_csv(input_file: str, output_csv: str):
    
    all_dataframes = []
    
    with open(input_file, 'r', encoding='utf-8') as file:
        data = file.read()
        
    sentences = data.strip().split('\n\n')
    
    for sentence in sentences:
        rows = sentence.strip().split('\n')
        columns = [row.split('\t')[:11] for  row in rows]
        df = pd.DataFrame(columns)
        df = df[~df[0].str.startswith("# newdoc id") & ~df[0].str.startswith("# sent_id") & ~df[0].str.startswith("# text")]
        
        num_verbs = df[3].apply(lambda x: 1 if x in ['AUX','VERB'] else 0).sum()
    
        for i in range(num_verbs):
            all_dataframes.append(df)
            
    big_df = pd.concat(all_dataframes, ignore_index=True)
    
    big_df.to_csv(output_csv)
    print(f'File {os.path.basename(output_csv)} produced!')
    

read_conllu_write_csv('data/input/en_ewt-up-test.conllu','data/output/test.csv')
read_conllu_write_csv('data/input/en_ewt-up-train.conllu','data/output/train.csv')