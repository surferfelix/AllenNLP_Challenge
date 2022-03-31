import csv
import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

def reader(input: str):
    parsed = []
    corrects = []
    golds = []
    with open(input) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            parse = row[0]
            correct = row[1]
            gold = row[-1]
            parsed.append(parse)
            corrects.append(correct)
            golds.append(gold)
    return parsed, corrects, golds


def evaluator(parsed, corrects, golds, path):
    '''Runs evaluation'''
    # Test identification
    preds = []
    name = path.split('/')[-1].rstrip('.csv')
    for i, (p, c, g) in enumerate(zip(parsed, corrects, golds)):
        predict_iterator = p.split('[')
        relevant_ind = 'undefined'
        for index, it in enumerate(predict_iterator): # Still going over every row
            if it.startswith(g.strip()):
                relevant_ind = index
        try:
            preds.append(predict_iterator[relevant_ind].split(':')[0].strip())
        except TypeError:
                if 'temporal_eval' in name:
                    preds.append(predict_iterator[-1].split(':')[0].strip()) # Likely models alternative pred
                elif 'name_eval' in name:
                    preds.append(predict_iterator[0].split(':')[0].strip()) # Likely models alternative pred
                elif 'instrument_eval' in name:
                    preds.append(predict_iterator[-1].split(':')[0].strip()) # Likely models alternative pred
                elif 'atypical_eval' in name:
                    preds.append(predict_iterator[0].split(':')[0].strip()) # Likely models alternative pred (Actually just checking the alternative pred in chunk position of target)
                elif 'instr_as_agent' in name:
                    preds.append(predict_iterator[0].split(':')[0].strip()) # Likely models alternative pred (Actually just checking the alternative pred in chunk position of target) 
                else:
                    preds.append(predict_iterator[0].split(':')[0].strip())
    return preds, golds, name

def reporting(preds, golds, name):
    golds = [gold.strip() for gold in golds]
    labels = sorted(set(preds), key=preds.index)
    confusion_report = confusion_matrix(y_true = golds, y_pred=preds, labels = labels)
    matrix = pd.DataFrame(confusion_report, index = labels, columns = labels)
    print(name)
    print(matrix.style.to_latex())
    matrix.to_csv(f'evaluations/{name}.csv')

def fails_rate(preds, golds, name):
    golds = [gold.strip() for gold in golds]
    print(f'ACCURACY EVAL: {name}')
    correct = 0
    for pred, gold in zip(preds, golds):
        if pred == gold:
            correct+=1
    accuracy = correct/len(preds)*100
    print(f'ACCURACY IS: {accuracy}')

def main():
    '''Runs the evaluation'''
    for path in os.listdir('output'):
        if path.endswith('csv'): # We want to avoid the summary which is txt
            parsed, corrects, golds = reader('output/'+path)
            preds,golds, name = evaluator(parsed, corrects, golds, path)
            reporting(preds, golds, name)
            fails_rate(preds, golds, name)

if __name__ == '__main__':
    main()