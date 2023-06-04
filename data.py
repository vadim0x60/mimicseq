from google.cloud import bigquery
from torch.utils.data import Dataset

client = bigquery.Client(project='graphsim')

class MIMICSEQ(Dataset):
    def __init__(self, hadm_ids, 
                 id_transform=lambda x: x,
                 intensity_transform=lambda x: x):
        self.hadm_ids = hadm_ids
        self.id_transform = id_transform
        self.intensity_transform = intensity_transform

    def __getitem__(self, index):
        hadm_id = self.hadm_ids[index]
        q = 'SELECT label, intensity FROM `graphsim.mimic.events` WHERE hadm_id = ' 
        q += str(hadm_id) + ' ORDER BY eventtime'
        history = client.query(q).to_dataframe()
        event_ids = self.id_transform(history['event_id'].tolist())
        intensities = self.intensity_transform(history['intensity'].tolist())
        return event_ids, intensities

    def __len__(self):
        return len(self.hadm_ids)

def load_fold(fold, *args, **kwargs):
    hadm_ids = client.query(f'SELECT * FROM graphsim.mimic.folds WHERE fold = "{fold}"')
    hadm_ids = hadm_ids.to_dataframe()['hadm_id'].tolist()
    return MIMICSEQ(hadm_ids, *args, **kwargs)

def load_mimicseq(id_transform=lambda x: x,
                  intensity_transform=lambda x: x):
    # Initialize the BigQuery client
    client = bigquery.Client(project='graphsim')
    legend = client.query('SELECT * FROM graphsim.mimic.eventtypes').to_dataframe()
    train_data = load_fold('train', id_transform, intensity_transform)
    test_data = load_fold('test', id_transform, intensity_transform)
    return legend, train_data, test_data
