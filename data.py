from google.cloud import bigquery
from torch.utils.data import Dataset

client = bigquery.Client(project='graphsim')

class MIMICSEQ(Dataset):
    def __init__(self, hadm_ids, 
                 transform=lambda x: x):
        self.hadm_ids = hadm_ids
        self.transform = transform

    def __getitem__(self, index):
        hadm_id = self.hadm_ids[index]
        q = 'SELECT event_id, intensity FROM `graphsim.mimic.events` WHERE hadm_id = ' 
        q += str(hadm_id) + ' ORDER BY eventtime'
        history = client.query(q).to_dataframe()
        
        event_ids = history['event_id'].tolist()
        intensities = history['intensity'].tolist()

        return self.transform(event_ids, intensities)

    def __len__(self):
        return len(self.hadm_ids)

def load_fold(fold, *args, **kwargs):
    hadm_ids = client.query(f'SELECT * FROM graphsim.mimic.folds WHERE fold = "{fold}"')
    hadm_ids = hadm_ids.to_dataframe()['hadm_id'].tolist()
    return MIMICSEQ(hadm_ids, *args, **kwargs)

def load_train(transform=lambda x: x):
    return load_fold('train', transform)

def load_test(transform=lambda x: x):
    return load_fold('test', transform)

def load_legend():
    q = 'SELECT * FROM graphsim.mimic.eventtypes ORDER BY event_id'
    return client.query(q).to_dataframe()
