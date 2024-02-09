import pandas as pd
from benchmarks import second_day_event_classification
from benchmarks import second_day_intensity_regression
import numpy.random as npr

def vibe_based_forecaster(eventtypes, prefixes):
    # Dummy model that predicts 10% of events
    return npr.choice([0, 1], size=len(eventtypes), p=[0.9, 0.1])
                      
def the_more_things_change_the_more_they_stay_the_same(eventtypes, prefixes):
    # Assome that the same events will happen on day n+1 as on days 1..n
    return eventtypes.index.isin(prefixes['event_id'])

def hopefully_intensity_is_low_variance(eventtypes, prefixes):
    # Assume that the intensity of the events will stay the same
    return eventtypes['intensity_mean']

if __name__ == '__main__':
    # Evaluate the dummy model on the test set
    benchmark = second_day_event_classification()

    for predictor in [vibe_based_forecaster, 
                      the_more_things_change_the_more_they_stay_the_same]:
        pred = predictor(benchmark.eventtypes, benchmark.prefix)
        precision, recall = benchmark.metrics(pred)
        print(f'Classification. Precision: {precision:.2f}, Recall: {recall:.2f}')

    #benchmark = second_day_intensity_regression()
    #pred = hopefully_intensity_is_low_variance(benchmark.eventtypes, benchmark.prefix)
    #r2 = benchmark.metrics(pred)
    #print(f'Regression. R2: {r2:.2f}')