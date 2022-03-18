# AllenNLP_Challenge
Creating challenge sets to challenge pre-trained AllenNLP SRL models with Checklist implementation       

## How the tests work

The tests are stored in the tests directory. Each file in this directory contains an experiment. Typically, the Checklist framework is used to enrich the test cases.

The content of _tests/_ is structured in the following way:
- `Someone killed {first_name} {last_name} last night , ARG1`
Where `{first_name}` and `{last_name}` are being used to generate test examples through Checklist. 
and where `ARG1` represents the gold label that they should correspond to. 

## How to run

Navigate to the AllenNLP_Challenge directory. 
`cd AllenNLP_Challenge`

Create a new anaconda environment (This requires you to install anaconda first)
`conda create --name myenv`

Switch to this conda environment
`conda activate myenv`

Install the dependencies with the following command
`pip install -r requirements.txt`

Run the program
`python allencheck.py`
