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
        infile = test_path.read()
    return infile

def preprocess():
    pass

def less_verbose():
    logging.getLogger('allennlp.common.params').disabled = True 
    logging.getLogger('allennlp.nn.initializers').disabled = True 
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO) 
    logging.getLogger('urllib3.connectionpool').disabled = True 


def load_predictions(text):
    srl_predictor = load_predictor('structured-prediction-srl')
    output = srl_predictor.predict(text)

def get_arg(pred, arg_target='ARG1'):
    # we assume one predicate:
    predicate_arguments = pred['verbs'][0]
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

def found_arg1_people(x, pred, conf, label=None, meta=None):
    
    # people should be recognized as arg1

    people = set([meta['first_name'], meta['last_name']])
    arg_1 = get_arg(pred, arg_target='ARG1')

    if arg_1 == people:
        pass_ = True
    else:
        pass_ = False
    return pass_

def main(case): 
    print('Initializing AllenNLP...')
    less_verbose()
    print(f'Reading test case {case}...')
    read_test(f'tests/{case}')
    


# Running the code from this file
if __name__ == '__main__':
    test_cases = os.listdir('tests')
    for case in test_cases:
        main(case)