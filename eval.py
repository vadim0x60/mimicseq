import numpy as np

def eval_event_pred(tails, next_event_pred):
    """
    Evaluate the event prediction accuracy of a model

    tails: batch_size x sequence_length, np array of event ids
    next_event_pred: batch_size, np array of event ids
    """

    hard_acc = (next_event_pred == tails[:,0]).astype(float).mean()
    soft_acc = np.isin(next_event_pred, tails).astype(float).mean()

    return hard_acc, soft_acc

def eval_intensity_pred(event_tails, intensity_tails, pred_intensities):
    """
    Evaluate the intensity prediction accuracy of a model

    event_tails: batch_size x sequence_length, np array of event ids
    intensity_tails: batch_size x sequence_length, np array of intensities
    pred_intensities: batch_size x n_event_types, np array of predicted intensities
    """

    result = 0

    for event_tail, intensity_tail, pred_int in zip(event_tails, intensity_tails, pred_intensities):
        event_tail, ids = np.unique(event_tail, return_index=True)
        intensity_tail = intensity_tail[ids]
        intensity_forecast = pred_int[event_tail]
        result -= (intensity_forecast - intensity_tail).norm() / intensity_tail.norm()

    return result / len(event_tails)