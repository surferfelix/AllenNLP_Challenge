import csv


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

def evaluator(parsed, corrects, golds):
    '''Runs evaluation'''
    # Test identification
    pass

def main(file_path):
    '''Runs the evaluation'''
    parsed, corrects, golds = reader(file_path)
    evaluator(parsed, corrects, golds)
if __name__ == '__main__':
    file_path = 'output/result.csv'
    main(file_path)