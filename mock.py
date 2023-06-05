from torch.utils.data import Dataset
import pandas as pd

legend = pd.DataFrame({
    'label': ['a', 'b', 'c', 'd'],
    'avg_intensity': [1, 1, 1, 1],
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
                 id_transform=lambda x: x,
                 intensity_transform=lambda x: x):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.id_transform = id_transform
        self.intensity_transform = intensity_transform

    def __getitem__(self, index):
        return self.id_transform(patients[self.start_idx + index]), self.intensity_transform(1)

    def __len__(self):
        return self.end_idx - self.start_idx

def load_fold(fold, *args, **kwargs):
    if fold == 'train':
        return MOCKSEQ(0, 9, *args, **kwargs)
    elif fold == 'test':
        return MOCKSEQ(9, 10, *args, **kwargs)

def load_mockseq(id_transform=lambda x: x,
                 intensity_transform=lambda x: x):
    train_data = load_fold('train', id_transform, intensity_transform)
    test_data = load_fold('test', id_transform, intensity_transform)
    return legend, train_data, test_data
