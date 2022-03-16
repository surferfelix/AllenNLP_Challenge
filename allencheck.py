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

def read_test(test_path):
    with open(test_path) as file:
        infile = file.read()
    return infile


def predict_srl(data):
    
    pred = []
    srl_predictor = load_predictor('structured-prediction-srl')
    for d in data:
        pred.append(srl_predictor.predict(d))
    return pred

def preprocess(infile):
    '''Target array consisting of ID's'''
    container = []
    for i in infile:
        case_dict = {}

def run_case(text):
    editor = Editor()
    expect_arg1 = Expect.single(found_arg1_people)
    predict_and_conf = PredictorWrapper.wrap_predict(predict_srl)
    t = editor.template("Someone killed {first_name} {last_name} last night.", meta = True, nsamples= 10)
    test = MFT(**t, expect=expect_arg1)
    test.run(predict_and_conf)
    test.summary(format_example_fn=format_srl)
    #  t = editor.template("Someone killed {first_name} {last_name} last night.", meta=True, nsamples=10)


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
    
    # people should be recognized as arg1

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

def main(case): 
    print('Initializing AllenNLP...')
    less_verbose()
    print(f'Reading test case {case}...')
    # a =read_test(f'tests/{case}')
    # print(a)
    run_case('hihi')
# Running the code from this file
if __name__ == '__main__':
    test_cases = os.listdir('tests')
    for case in test_cases:
        main(case)