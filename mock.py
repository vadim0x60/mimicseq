from torch.utils.data import Dataset
import pandas as pd

legend = pd.DataFrame({
    'label': ['a', 'b', 'c', 'd'],
    'intensity_mean': [1, 1, 1, 1],
    'intensity_std': [1, 1, 1, 1],
    'event_id': [1, 2, 3, 4]
})
patients = [
    [1],
    [2],
    [3],
    [4],
    [1, 2],
    [2, 3],
    [3, 4],
    [1, 2, 3],
    [2, 3, 4],
    [1, 2, 3, 4]
]

class MOCKSEQ(Dataset):
    def __init__(self, start_idx, end_idx, 
                 transform=lambda x: x):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(patients[self.start_idx + index], 1)
   
    def __len__(self):
        return self.end_idx - self.start_idx

def load_fold(fold, *args, **kwargs):
    if fold == 'train':
        return MOCKSEQ(0, 9, *args, **kwargs)
    elif fold == 'test':
        return MOCKSEQ(9, 10, *args, **kwargs)
    
def load_train(transform=lambda x: x):
    return MOCKSEQ(0, 9, transform)

def load_test(transform=lambda x: x):
    return MOCKSEQ(9, 10, transform)

def load_legend():
    return legend