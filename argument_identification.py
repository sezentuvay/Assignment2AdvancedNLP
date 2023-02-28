import pandas as pd

def make_gold_binary(inputfile, outputfile):
    '''
    Function that takes the inputfile (which is the output of feature_extraction.py), and changes the numbers in the last column to zeros and ones
    
    :param inptufile: path to the inputfile
    :param outputfile: path to the outputfile
    
    :type outputfile: string
    '''
    
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
    inputfiles = ['TEST_CSV.csv', 'TRAIN_CSV.csv', 'DEV_CSV.csv']
    for inputfile in inputfiles:
        make_gold_binary('data/output/'+inputfile, 'data/output/bin_'+inputfile)

if __name__ == '__main__':
    main()

