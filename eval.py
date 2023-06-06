import numpy as np
from sklearn.metrics import r2_score

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

    event_tails = np.array(event_tails)
    intensity_tails = np.array(intensity_tails)
    pred_intensities = np.array(pred_intensities)

    for event_tail, intensity_tail, pred_int in zip(event_tails, intensity_tails, pred_intensities):
        # Remove duplicate events
        # When predicting, say, blood pressure, we want to predict the soonest reading
        event_tail, ids = np.unique(event_tail, return_index=True)
        intensity_tail = intensity_tail[ids]
        
        # Remove events that don't have an intensity
        no_intensity = np.isnan(intensity_tail)
        event_tail = event_tail[~no_intensity]
        intensity_tail = intensity_tail[~no_intensity]

        intensity_forecast = pred_int[event_tail]
        result += r2_score(intensity_tail, intensity_forecast)
        
    return result / len(event_tails)