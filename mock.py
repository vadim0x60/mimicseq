from torch.utils.data import Dataset
import pandas as pd

legend = pd.DataFrame({
    'label': ['a', 'b', 'c', 'd'],
    'intensity': [1, 1, 1, 1],
    'event_id': [1, 2, 3, 4]
})
patients = [
    [1, 2, 3, 4],
    [4, 1, 2, 3],
    [3, 4, 1, 2],
    [2, 3, 4, 1]
]

class MOCKSEQ(Dataset):
    def __init__(self, start_idx, end_idx):
        self.start_idx = start_idx
        self.end_idx = end_idx

    def __getitem__(self, index):
        return patients[self.start_idx + index], 1

    def __len__(self):
        return self.end_idx - self.start_idx

def load_fold(fold):
    if fold == 'train':
        return MOCKSEQ(0, 3)
    elif fold == 'test':
        return MOCKSEQ(3, 4)

def load_mockseq():
    train_data = load_fold('train')
    test_data = load_fold('test')
    return legend, train_data, test_data
