from sklearn.metrics import f1_score
from sklearn.metrics import r2_score


def score(predict, target, type='f1'):
    """Evaluate accuracy.
    
    predict: predict lables
    target: target labels
    """
    if type is 'f1':
        score = f1_score(target.values.astype(int), predict.values.astype(float), average='micro')
    elif type is 'r2':
        score = r2_score(target.values, predict.values)
    else:
        raise ValueError('the score type is wrong. please choice f1 or r2.')

    return score
