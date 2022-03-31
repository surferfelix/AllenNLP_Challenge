from allennlp_models.pretrained import load_predictor
import checklist
from checklist.editor import Editor
from checklist.perturb import Perturb
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect
import allennlp_models
from checklist.pred_wrapper import PredictorWrapper
import logging
import os
import shutil
import tempfile
from unittest import TestCase
from allennlp.common.checks import log_pytorch_version_info
import os
import csv
import sys

def read_test(test_path):
    container = []
    with open(test_path) as file:
        infile = csv.reader(file, delimiter = ',', quotechar = '|')
        for index, row in enumerate(infile):
            if not index == 0: # Skipping headers
                container.append(row)

    return container

def preprocess(infile):
    ''''''
    inputs = []
    golds = []
    for row in infile:
        inp = row[0]
        gold = row[1]
        inputs.append(inp)
        golds.append(gold)
    return inputs, golds

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


def less_verbose():
    logging.getLogger('allennlp.common.params').disabled = True 
    logging.getLogger('allennlp.nn.initializers').disabled = True 
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO) 
    logging.getLogger('urllib3.connectionpool').disabled = True 

# def load_predictions(text):
#     srl_predictor = load_predictor('structured-prediction-srl')
#     output = srl_predictor.predict(text)

## These are the two core-checklist functions
def get_arg(pred, arg_target='ARG1'):
    predicate_arguments = pred['verbs'][0] # we assume one predicate:
    words = pred['words']
    tags = predicate_arguments['tags']
    arg_list = []
    for t, w in zip(tags, words):
        arg = t
        if '-' in t:
            arg = t.split('-')[1]
            print(arg)
        if arg == arg_target:
            arg_list.append(w)
    arg_set = set(arg_list)
    return arg_set

def get_argm(pred, arg_target='ARG1'):
    predicate_arguments = pred['verbs'][0] # we assume one predicate:
    words = pred['words']
    tags = predicate_arguments['tags']
    arg_list = []
    for t, w in zip(tags, words):
        arg = t
        if '-' in t:
            arg = t.split('-')[-1]
            print(arg)
        if arg == arg_target:
            arg_list.append(w)
    arg_set = set(arg_list)
    return arg_set


def format_srl(x, pred, conf, label=None, meta=None):
    '''Helper function to display failures'''
    results = []
    predicate_structure = pred['verbs'][0]['description']
        
    return predicate_structure
##

# The following functions can tell the model what to expect

# Testing with people names
def found_arg1_people(x, pred, conf, label=None, meta=None):
    
    # Setting the gold label

    people = set([meta['first_name'], meta['last_name']])
    arg_1 = get_arg(pred, arg_target='ARG1')

    if arg_1 == people:
        pass_ = True
    else:
        pass_ = False
    return pass_

# Testing arg2 with instrument
def found_arg2_instrument(x, pred, conf, label=None, meta=None):
    
    # people should be recognized as arg1
    
    instrument = set(meta['instrument'].split(' '))
    arg_3 = get_arg(pred, arg_target='ARG2')

    if arg_3 == instrument:
        pass_ = True
    else:
        pass_ = False
    return pass_

def found_arg2_tool(x, pred, conf, label=None, meta=None):
    
    # people should be recognized as arg1
    
    instrument = set(meta['tool'].split(' '))
    arg_3 = get_arg(pred, arg_target='ARG2')

    if arg_3 == instrument:
        pass_ = True
    else:
        pass_ = False
    return pass_

def found_atypical_arg_0(x , pred, conf, label = None, meta = None):
    a_arg = set(meta['atypical'].split(' '))
    system_pred = get_arg(pred, arg_target = 'ARG0')
    if a_arg == system_pred:
        pass_ = True
    else:
        pass_ = False
    return pass_

def found_temp_argm(x, pred, conf, label = None, meta = None):
    a_arg = set(meta['temporal'].split(' '))
    system_pred = get_argm(pred, arg_target = 'TMP')
    if a_arg == system_pred:
        pass_ = True
    else:
        pass_ = False
    return pass_

def run_case(text, gold, index, model):
    '''Will run the experiment for a specific example
    :param text: The text with appropiate label
    :param gold: pointer for the gold label corresponding to the input'''
    print('Inside run_case function')
    editor = Editor() # Initializing Editor object
    # This is where we try to identify what to evaluate here
    if "{first_name}" in text and 'ARG1' in gold:
        print('names test')
        expectation = Expect.single(found_arg1_people) # Specify what case should expect
        if model == 'BERT':
            predict_and_conf = PredictorWrapper.wrap_predict(predict_srl_bert) # Wrap the prediction in checklist format
        elif model == 'Bi-LSTM':
            predict_and_conf = PredictorWrapper.wrap_predict(predict_srl) # Wrap the prediction in checklist format
        t = editor.template(text, meta = True, nsamples= 30) # The case to run
        test = MFT(**t, expect=expectation)
        test.run(predict_and_conf)
        write_out_json(test.results, index, gold, f'name_eval_{model}.csv')
    elif "{instrument}" in text and "ARG2" in gold:
        print('instr test')
        instruments = ['with a spoon', 'with a fork', 'with a knife', 'with a pinecone', 'with a plate', 'with a candle', 'with a spork', 'using a knife', 'using a plate', 'using a cup']
        expectation = Expect.single(found_arg2_instrument)
        if model == 'BERT':
            predict_and_conf = PredictorWrapper.wrap_predict(predict_srl_bert) # Wrap the prediction in checklist format
        elif model == 'Bi-LSTM':
            predict_and_conf = PredictorWrapper.wrap_predict(predict_srl) # Wrap the prediction in checklist format
        t = editor.template(text, instrument = instruments, meta = True, nsamples= 30) # The case to run
        test = MFT(**t, expect=expectation)
        test.run(predict_and_conf)
        write_out_json(test.results, index, gold, f'instrument_eval_{model}.csv')
    elif "{atypical}" in text and "ARG0" in gold:
        print('atyp test')
        atypicals = ['John', 'Mary', "A dog", "A book", "A ship", 'a rocket', 'the toothbrush', 'the carrot', 'africa', 'senegal', 'the street', 'the glasses']
        expectation = Expect.single(found_atypical_arg_0)
        if model == 'BERT':
            predict_and_conf = PredictorWrapper.wrap_predict(predict_srl_bert) # Wrap the prediction in checklist format
        elif model == 'Bi-LSTM':
            predict_and_conf = PredictorWrapper.wrap_predict(predict_srl) # Wrap the prediction in checklist format
        t = editor.template(text, atypical = atypicals, meta = True, nsamples= 30) # The case to run
        test = MFT(**t, expect=expectation)
        test.run(predict_and_conf)
        write_out_json(test.results, index, gold, f'atypical_eval_{model}.csv')
    elif "{temporal}" in text and 'ARGM-TMP' in gold:
        print('temporal test')
        temporals = ['tomorrow', 'in an hour', 'in a bit', 'soon', 'in a while', 'next month', 'next year',
        'at 12', 'at noon', 'tonight', 'tomorrow morning']
        expectation = Expect.single(found_temp_argm)
        if model == 'BERT':
            predict_and_conf = PredictorWrapper.wrap_predict(predict_srl_bert) # Wrap the prediction in checklist format
        elif model == 'Bi-LSTM':
            predict_and_conf = PredictorWrapper.wrap_predict(predict_srl) # Wrap the prediction in checklist format
        t = editor.template(text, temporal = temporals, meta = True, nsamples= 30) # The case to run
        test = MFT(**t, expect=expectation)
        test.run(predict_and_conf)
        write_out_json(test.results, index, gold, f'temporal_eval_{model}.csv')
    # elif "{tool}" in text and "ARG2" in gold:
    #     print('instr test')
    #     tools = ['a spoon', 'a fork', 'a knife', 'an axe', 'a plate', 'a candle', 'a spork', 'cutlery', 'a phone', 'a blade', 'a machete']
    #     expectation = Expect.single(found_arg2_tool)
    #     t = editor.template(text, tool = tools, meta = True, nsamples= 30) # The case to run
    #     if model == 'BERT':
    #         predict_and_conf = PredictorWrapper.wrap_predict(predict_srl_bert) # Wrap the prediction in checklist format
    #     elif model == 'Bi-LSTM':
    #         predict_and_conf = PredictorWrapper.wrap_predict(predict_srl) # Wrap the prediction in checklist format
    #     test = MFT(**t, expect=expectation)
    #     test.run(predict_and_conf)
    #     write_out_json(test.results, index, gold, f'instrument_as_arg_eval_{model}.csv')
    else:
        return "oops, no implementation possible yet for this kind of data :("
    if index == 0:
        # Print to file trick taken from https://howtodoinjava.com/examples/python-print-to-file/
        original_stdout = sys.stdout # Saving original state
        with open("output/raw_output.txt", 'w') as output: # Overwrite contents on fitst iteration
            sys.stdout = output # Changing state
            print('GOLD: '+ gold)
            print(test.summary(format_example_fn=format_srl))
            sys.stdout = original_stdout 
    elif index > 0: 
        original_stdout = sys.stdout # Saving original state
        with open("output/raw_output.txt", 'a') as output: # Append on following iterations
            sys.stdout = output # Changing state
            print('GOLD: '+ gold)
            print(test.summary(format_example_fn=format_srl))
            sys.stdout = original_stdout 
    for i, case in enumerate(test.data):
        print(i, case)

def write_out_json(results, index, gold: str, output_file_name):
    ''':param gold: This is the gold label for the iteration'''
    predictions = results['preds']
    print(predictions)
    answers = results['passed']
    for p, a in zip(predictions, answers):
        print(p['verbs'][0]['description'], a)
    if index == 0:
        with open(f'output/{output_file_name}', 'w') as txt:
            writer = csv.writer(txt)
            writer.writerow(['INPUT', 'EVAL', 'GOLD'])
            for p, a in zip(predictions, answers):
                writer.writerow([p['verbs'][0]['description'],a, gold])
    elif index > 0:
        with open(f'output/{output_file_name}', 'a') as txt:
            writer = csv.writer(txt)
            for p, a in zip(predictions, answers):
                writer.writerow([p['verbs'][0]['description'],a, gold])

def main(case, file_nr, model): 
    '''This main function iterates and runs cases for each line in the CSV input'''
    print('Initializing AllenNLP...')
    less_verbose()
    print(f'MODEL: RUNNING CURRENT CASE: {case}\t...')
    text = read_test(f'tests/{case}') # Outputs the lines
    print('full text:', text)
    inps, golds = preprocess(text) # tuple inputs, golds
    print('preprocessed text:', text)
    for index, (inp, gold) in enumerate(zip(inps, golds)):
        if file_nr > 0:
            print('WORKING')
            index += 1 # Really doesn't matter what index is since this program has boolean logic
        run_case(inp, gold, index, model)

# Running the code from this file
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