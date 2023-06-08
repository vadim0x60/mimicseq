from google.cloud import bigquery
from torch.utils.data import Dataset, Sampler

client = bigquery.Client(project='graphsim')

class BatchSamplerForPaddingHaters(Sampler):
    def __init__(self, lengths, sampler):
        self.sampler = sampler
        self.lengths = lengths
        self.batch_sizes = []

        cur_len = lengths[0]
        cur_bs = 0
        for l in lengths:
            if l != cur_len:
                self.batch_sizes.append(cur_bs)
                cur_bs = 0
            cur_bs += 1
            cur_len = l

    def __iter__(self):
        sampler_iter = iter(self.sampler)
        for bs in self.batch_sizes:
            yield [next(sampler_iter) for _ in range(bs)]

    def __len__(self):
        return len(self.batch_sizes)

class MIMICSEQ(Dataset):
    def __init__(self, hadm_ids, lengths,
                 transform=lambda x: x):
        self.hadm_ids = hadm_ids
        self.lengths = lengths
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
    q = f'SELECT * FROM graphsim.mimic.folds WHERE fold = "{fold}" ORDER BY len'
    folds = client.query(q).to_dataframe()

    return MIMICSEQ(folds['hadm_id'].tolist(), folds['len'].tolist(), *args, **kwargs)

def load_train(transform=lambda x: x):
    return load_fold('train', transform)

def load_test(transform=lambda x: x):
    return load_fold('test', transform)

def load_legend():
    q = 'SELECT * FROM graphsim.mimic.eventtypes ORDER BY event_id'
    return client.query(q).to_dataframe()
