def dummy_model(leadup):
    """History tends to repeat itself, innit?"""
    prediction = leadup.copy()
    prediction['eventtime'] += prediction['eventtime'].max() - prediction['eventtime'].min()
    return prediction

if __name__ == '__main__':
    from evaluate import evaluate_model
    import pandas as pd

    test_data = pd.read_parquet('data/test.parquet').groupby('sample_id')
    episodes = (episode for idx, episode in test_data)

    evaluate_model(dummy_model, episodes, callback=print)