from google.cloud import bigquery
from torch.utils.data import Dataset

client = bigquery.Client(project='graphsim')

class MIMICSEQ(Dataset):
    def __init__(self, hadm_ids):
        self.hadm_ids = hadm_ids

    def __getitem__(self, index):
        hadm_id = self.hadm_ids[index]
        q = 'SELECT label, intensity FROM `graphsim.mimic.events` WHERE hadm_id = ' 
        q += str(hadm_id) + ' ORDER BY eventtime'
        history = client.query(q).to_dataframe()
        return history['event_id'], history['intensity']

    def __len__(self):
        return len(self.hadm_ids)

def load_fold(fold):
    hadm_ids = client.query(f'SELECT * FROM graphsim.mimic.folds WHERE fold = "{fold}"')
    hadm_ids = hadm_ids.to_dataframe()['hadm_id'].tolist()
    return MIMICSEQ(hadm_ids)

def load_mimicseq():
    # Initialize the BigQuery client
    client = bigquery.Client(project='graphsim')
    legend = client.query('SELECT * FROM graphsim.mimic.eventtypes').to_dataframe()
    train_data = load_fold('train')
    test_data = load_fold('test')
    return legend, train_data, test_data
