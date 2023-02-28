import pandas as pd

def make_gold_binary(inputfile, outputfile):
    data = pd.read_csv(inputfile, encoding = 'utf-8', sep = '\t')
    binary_gold = []
    for gold in data['Gold']:
        if str(gold).startswith('ARG'):
            binary_gold.append(1)
        else:
            binary_gold.append(0)

    data.drop('Gold',axis=1)
    data['Gold'] = binary_gold
    data.to_csv(outputfile)

def main():
    inputfiles = ['test_new.csv', 'dev_new.csv', 'train_new.csv']
    for inputfile in inputfiles:
        make_gold_binary('data/output/'+inputfile, 'data/bin_'+inputfile)

if __name__ == '__main__':
    main()



