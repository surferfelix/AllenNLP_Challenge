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
        infile = csv.reader(file, delimiter = ',')
        for index, row in enumerate(infile):
            if not index == 0: # Skipping headers
                container.append(row)

    return container

def predict_srl(data):
    pred = []
    srl_predictor = load_predictor('structured-prediction-srl')
    for d in data:
        pred.append(srl_predictor.predict(d))
    return pred

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

def less_verbose():
    logging.getLogger('allennlp.common.params').disabled = True 
    logging.getLogger('allennlp.nn.initializers').disabled = True 
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO) 
    logging.getLogger('urllib3.connectionpool').disabled = True 

def load_predictions(text):
    srl_predictor = load_predictor('structured-prediction-srl')
    output = srl_predictor.predict(text)

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
    system_pred = get_arg(pred, arg_target = 'ARGMNR')
    if a_arg == system_pred:
        pass_ = True
    else:
        pass_ = False
    return pass_

def run_case(text, gold, index):
    '''Will run the experiment for a specific example
    :param text: The text with appropiate label
    :param gold: pointer for the gold label corresponding to the input'''
    editor = Editor() # Initializing Editor object
    # This is where we try to identify what to evaluate here
    if "{first_name}" in text and 'ARG1' in gold:
        expectation = Expect.single(found_arg1_people) # Specify what case should expect
        predict_and_conf = PredictorWrapper.wrap_predict(predict_srl) # Wrap the prediction in checklist format
        t = editor.template(text, meta = True, nsamples= 30) # The case to run
    elif "{instrument}" in text and "ARG2" in gold:
        instruments = ['with a spoon', 'with a fork', 'with a knife', 'with a pinecone']
        expectation = Expect.single(found_arg2_instrument)
        predict_and_conf = PredictorWrapper.wrap_predict(predict_srl)
        t = editor.template(text, instrument = instruments, meta = True, nsamples= 30) # The case to run
    elif "{atypical}" in text and "ARG0" in gold:
        atypicals = ['John', 'Mary', "A dog", "A book", "A ship"]
        expectation = Expect.single(found_atypical_arg_0)
        predict_and_conf = PredictorWrapper.wrap_predict(predict_srl) # Wrap the prediction in checklist format
        t = editor.template(text, atypical = atypicals, meta = True, nsamples= 30) # The case to run
    elif "{temporal}" in text and 'ARGM' in gold:
        temporal_future = ['tomorrow', 'in an hour', 'in a bit', 'soon', 'in a while', 'next month', 'next year']
        expectation = Expect.single(found_temp_argm)
        predict_and_conf = PredictorWrapper.wrap_predict(predict_srl) # Wrap the prediction in checklist format
        t = editor.template(text, temporal = temporal_future, meta = True, nsamples= 30) # The case to run
    else:
        return "oops, no implementation possible yet for this kind of data :("
    test = MFT(**t, expect=expectation)
    test.run(predict_and_conf)
    write_out_json(test.results, index)
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

def write_out_json(results, index):
    predictions = results['preds']
    answers = results['passed']
    for p, a in zip(predictions, answers):
        print(p['words'], a)
    if index == 0:
        with open('output/result.csv', 'w') as txt:
            writer = csv.writer(txt)
            writer.writerow(['INPUT', 'EVAL'])
            for p, a in zip(predictions, answers):
                writer.writerow([p['words'],a])
    elif index > 0:
        with open('output/result.csv', 'a') as txt:
            writer = csv.writer(txt)
            for p, a in zip(predictions, answers):
                writer.writerow([p['words'],a])

def main(case, file_nr): 
    '''This main function iterates and runs cases for each line in the CSV input'''
    print('Initializing AllenNLP...')
    less_verbose()
    print(f'MODEL: RUNNING CURRENT CASE: {case}\t...')
    text = read_test(f'tests/{case}') # Outputs the lines
    inps, golds = preprocess(text) # tuple inputs, golds
    for index, (inp, gold) in enumerate(zip(inps, golds)):
        if file_nr > 0:
            index += 1 # Really doesn't matter what index is since this program has boolean logic
        run_case(inp, gold, index)

# Running the code from this file
if __name__ == '__main__':
    test_cases = os.listdir('tests')
    for index, case in enumerate(test_cases):
        file_nr = index
        if not case.startswith('.'): # Omitting dotfiles
            main(case, file_nr)