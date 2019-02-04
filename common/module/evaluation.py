from sklearn.metrics import f1_score
from termcolor import cprint


def score(predict, target):
    """Evaluate accuracy by using f1 score.
    
    predict: predict lables
    target: target labels
    """
    score = f1_score(target.values.astype(int), predict.values.astype(float), average='micro')

    passing_score = 0.80
    if score > passing_score:
        is_passed = True
    else:
        is_passed = False

    cprint('result: {}\nscore: {}'.format(is_passed, score), 'red', attrs=['bold'])