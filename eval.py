import numpy as np

def event_accuracy(tails, next_event_pred):
    """
    Evaluate the event prediction accuracy of a predictor on a sequence.

    tails: batch_size x sequence_length, np array of event ids
    next_event_pred: batch_size, np array of event ids
    """

    hard_acc = (next_event_pred == tails[:,0]).astype(float).mean()
    soft_acc = np.isin(next_event_pred, tails).astype(float).mean()

    return hard_acc, soft_acc