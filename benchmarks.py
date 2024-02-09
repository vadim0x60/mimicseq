import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

etpath = 'data/eventtypes.parquet'
gtpath = 'data/test.parquet'

class Benchmark:
    def __init__(self, target_day, eventtypes=None, ground_truth=None):
        gt = ground_truth
        et = eventtypes

        et = et if et is not None else pd.read_parquet(etpath)
        gt = gt if gt is not None else pd.read_parquet(gtpath, 
                                                       index='sample_id')
        
        gt = gt[gt['event_id'].isin(et.index)]
        et['event_id'] = et.index 

        start = gt.groupby('sample_id')['eventtime'].transform('min')
        end = gt.groupby('sample_id')['eventtime'].transform('max')

        if target_day > 0:
            gt['cutoff'] = start + pd.Timedelta(days=target_day)
        else:
            gt['cutoff'] = end - pd.Timedelta(days=-target_day)

        self.eventtypes = et
        self.prefix = gt[gt['eventtime'] < gt['cutoff']]
        self.target = gt[gt['eventtime'] > gt['cutoff']]

class EventBenchmark(Benchmark):
    def __init__(self, target_day, eventtypes=None, ground_truth=None, 
                 clustering='event_id'):
        super().__init__(target_day, eventtypes, ground_truth)

    def metrics(self, pred, episodes=None):
        episodes = episodes if episodes is not None else self.target['sample_id']
        true = self.target[self.target['sample_id'].isin(episodes)]

        tp = (pred * true).sum()
        precision = tp / pred.sum()
        recall = tp / true.sum()
        return (precision, recall)

class IntensityBenchmark(Benchmark):
    def __init__(self, target_day, eventtypes=None, ground_truth=None):
        eventtypes = eventtypes or pd.read_parquet('data/eventtypes.parquet')
        eventtypes = eventtypes[eventtypes['intensity_mean'] != 0]
        super().__init__(target_day, eventtypes, ground_truth)
        self.true = np.array([
            [self.target[self.target['event_id'] == e]['intensity'].mean()
             for e in self.eventtypes.index]
            for s in self.target['sample_id']
        ])

    def metrics(self, pred):
        return r2_score(self.true, pred)

def second_day_event_classification(clustering='event_id'):
    return EventBenchmark(2, clustering=clustering)

def last_day_event_classification(clustering='event_id'):
    return EventBenchmark(-1, clustering=clustering)

def second_day_intensity_regression():
    return IntensityBenchmark(2)

def last_day_intensity_regression():
    return IntensityBenchmark(-1)