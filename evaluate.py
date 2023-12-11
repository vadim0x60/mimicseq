import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from typing import Callable, Iterable

metrics = ['precision', 'recall', 'intensity_predicted', 'intensity_r2', 'time_predicted', 'time_r2']
clusterings = ['c10', 'c100', 'c1000', 'c10000', 'event_id']

eventtypes = pd.read_parquet('data/eventtypes.parquet')
eventtypes['event_id'] = eventtypes.index 

def annotate_clusters(episode):
    if 'event_id' in episode.columns and not episode.empty:
        # The data contains precise event information
        episode[clusterings] = episode['event_id'].apply(lambda x: eventtypes.loc[x][clusterings])
    else:
        # The data contains approximate event information (via event clusters)
        pass

def evaluate_model(model_predict: Callable[[pd.DataFrame], pd.DataFrame], 
                   episodes: Iterable[pd.DataFrame], 
                   callback: Callable[[pd.DataFrame], None] = lambda x: None):
    """
    Evaluate a predictive model on a dataset of episodes.
    
    Parameters
    ----------
    model_predict : a function that takes the beginning of the episode and predicts the rest
    episodes : an iterable of test episodes
    callback (optional): a logging function for each local evaluation
    
    Returns
    -------
    Global evaluation - average of local evaluations
    pd.DataFrame with columns ['precision', 'recall', 'intensity_r2', 'time_r2']
    and an index of the event clusterings
    Contains the evaluation metrics for each event clustering
    """

    index = pd.Index(clusterings, name='clustering')
    final_evaluation = pd.DataFrame(columns=metrics, index=index)
    count = 0

    for episode in episodes:
        episode = episode.sort_values('eventtime')
        annotate_clusters(episode)

        for i in range(len(episode)):
            leadup = episode.iloc[:i]
            truth = episode.iloc[i:]

            prediction = model_predict(leadup)
            annotate_clusters(prediction)

            evaluation = evaluate_prediction(prediction, truth)
            callback(evaluation)
            
            final_evaluation += evaluation
            count += 1

    final_evaluation /= count
    return final_evaluation

def evaluate_prediction(prediction: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Evaluate each metric for one predicted episode tail.
    
    Parameters
    ----------
    prediction : model's prediction of the future sequence of events
    pd.DataFrame with columns ['eventtime', 'intensity']
    + an event id column for every event clustering

    truth : the actual future sequence of events
    pd.DataFrame with columns ['eventtime', 'intensity']
    + an event id column for every event clustering

    Returns
    -------
    pd.DataFrame with columns ['precision', 'recall', 'intensity_r2', 'time_r2']
    and an index of the event clusterings
    Contains the evaluation metrics for each event clustering
    """

    index = pd.Index(clusterings, name='clustering')
    result = pd.DataFrame(columns=metrics, index=index)

    for clustering in clusterings:
        prediction = prediction.rename(columns={'eventtime': 'eventtime_pred'})
        truth = truth.rename(columns={'eventtime': 'eventtime_true'})

        if prediction.empty and truth.empty:
            precision = 1
            recall = 1
            intensity_predicted = None
            time_predicted = None
            intensity_r2 = None
            time_r2 = None
        elif prediction.empty or truth.empty:
            precision = 0
            recall = 0
            intensity_predicted = None
            time_predicted = None
            intensity_r2 = None
            time_r2 = None
        else:
            predvstrue = pd.merge_asof(prediction, truth, 
                                    left_on='eventtime_pred',
                                    right_on='eventtime_true', 
                                    by=clustering,
                                    suffixes=('_pred', '_true'))

            precision = len(predvstrue[clustering + '_pred'].dropna()) / len(prediction)
            recall = len(predvstrue[clustering + '_pred'].dropna()) / len(truth)

            with_intensity = predvstrue['intensity_pred'].notna() & predvstrue['intensity_true'].notna()
            intensities_count = predvstrue['intensity_true'].notna().sum() 
            intensity_predicted = with_intensity.sum() / intensities_count if intensities_count else None
            
            try:
                intensity_r2 = r2_score(predvstrue['intensity_pred'][with_intensity], 
                                        predvstrue['intensity_true'][with_intensity])
            except ValueError:
                intensity_r2 = None
            
            with_time = predvstrue['eventtime_pred'].notna()
            time_predicted = with_time.sum() / len(predvstrue['eventtime_true'])

            try:
                time_r2 = r2_score(predvstrue['eventtime_pred'][with_time].astype(np.int64),
                                   predvstrue['eventtime_true'][with_time].astype(np.int64))
            except ValueError:
                time_r2 = None
        
        result.loc[clustering] = [precision, recall, intensity_predicted, intensity_r2, time_predicted, time_r2]

    return result