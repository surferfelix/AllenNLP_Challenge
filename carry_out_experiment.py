import allencheck
import evaluate
import sys


def main():
    x = input('carry out experiment? Type y or n')
    if x.lower() == 'y':        
        allencheck.main()
    else:
        s = input('did not carry out experiment')
    z = input('carry out evaluation? Type y or n')
    if z.lower() == 'y':
        evaluate.main()
    else:
        print('did not carry out evaluation')

if __name__ == '__main__':
    main()