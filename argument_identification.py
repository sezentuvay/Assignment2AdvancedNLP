import pandas as pd

def make_gold_binary(inputfile, outputfile):
    #take the inputfile (which is the output of feature_extraction.py), and changes the numbers in the last column to zeros and ones
    #this is then saved in a new output file
    data = pd.read_csv(inputfile, encoding = 'utf-8', sep = '\t')
    binary_gold = []
    for gold in data['Gold']:
        if 'ARG' in str(gold):
            binary_gold.append(1)
        else:
            binary_gold.append(0)

    data.drop('Gold',axis=1)
    data['Gold'] = binary_gold
    data.to_csv(outputfile)

def main():
    inputfiles = ['test_new.csv', 'dev_new.csv', 'train_new.csv']
    for inputfile in inputfiles:
        make_gold_binary('data/output/'+inputfile, 'data/output/bin_'+inputfile)

if __name__ == '__main__':
    main()

#this script only needs to be run ones and all three files will be created. In the data-folder, you will need to create an output folder. 


