import conllu
import pandas as pd
import os
import pyarrow.feather as feather

def read_conllu_write_csv(input_file: str, output_csv: str):
    '''
    Reads data from a CoNLL-U format file, extracts specific columns, and writes them into a CSV file. 
    The function returns a pandas DataFrame containing the extracted data.

    Parameters:
    - input_file: A string representing the path to the input file in CoNLL-U format.
    - output_csv: A string representing the path to the output file in CSV format.

    Returns:
    - A pandas DataFrame containing the extracted data.

    '''
    #Read conllu file
    with open(input_file, 'r', encoding='utf-8') as file:
        data = file.read()
        
    sentences = data.strip().split('\n\n')  
    
    #List to store all dataframes
    all = []

    for sentence in sentences:
        rows = sentence.strip().split('\n')
        rows = [row for row in rows if not row[0].startswith('#')]
        columns = [row.split('\t') for row in rows]
        
        #Build dataframes only when we have more than 10 columns
        if len(columns[0])>10:
            df =pd.DataFrame(columns)  
            #Check number of predicates
            num_predicates = df[10].apply(lambda x: 1 if x !='_' else 0).sum()
            
            #Check if dataframe is in the correct form
            if num_predicates == df.shape[1]-11:
                #Build new dataframes with each predicate and append them to the original list
                for j in range(num_predicates):
                    
                    df_new = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,10+j+1]]
                    new_names = {col: k for k, col in enumerate(df_new.columns)}
                    df_new = df_new.rename(columns=new_names)
                    
                    all.append(df_new)
        

    big_df = pd.concat(all, ignore_index=True)
    big_df.to_csv(output_csv)
    print(f'File {os.path.basename(output_csv)} produced!')
    return big_df


def main():
    all_inputfiles = ['dev', 'train', 'test']

    for file in all_inputfiles:
        read_conllu_write_csv('data/input/en_ewt-up-'+file+'.conllu', 'data/output/'+file+'.csv')

if __name__ == '__main__':
    main()
