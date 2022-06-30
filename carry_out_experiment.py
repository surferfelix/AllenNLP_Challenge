import allencheck
import evaluate
import sys
import argparse
import os

def predict_srl(data):
    pred = []
    srl_predictor = load_predictor('structured-prediction-srl')
    for d in data:
        pred.append(srl_predictor.predict(d))
    return pred

def predict_srl_bert(data):
    pred = []
    srl_predictor = load_predictor('structured-prediction-srl-bert')
    for d in data:
        pred.append(srl_predictor.predict(d))
    return pred

def main(case, file_nr, model):
    x = input('carry out experiment? Type y or n')
    if x.lower() == 'y':        
        allencheck.main(case, file_nr, model)
    else:
        s = input('did not carry out experiment')
    z = input('carry out evaluation? Type y or n')
    if z.lower() == 'y':
        evaluate.main()
    else:
        print('did not carry out evaluation')

if __name__ == '__main__':
    models = [predict_srl, predict_srl_bert]
    test_cases = os.listdir('tests')
    for index, model in enumerate(models):
        if index == 0:
            model = 'Bi-LSTM'
        elif index == 1:
            model = 'BERT'
        for index, case in enumerate(test_cases):
            file_nr = index
            if not case.startswith('.') and case.endswith('.csv'): # Omitting dotfiles
                main(case, file_nr, model)