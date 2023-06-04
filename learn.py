from model import TimeSeriesTransformer
from data import load_mimicseq
import torch
import lightning as L
from initemb import init_embedding
import click
from mock import load_mockseq

@click.command()
@click.option('--real-data/--mock-data', default=True)
def train_icat(real_data):
    if real_data:
        legend, train_data, test_data = load_mimicseq(torch.LongTensor, torch.Tensor)
        initial_embedding = init_embedding(legend)
    else:
        legend, train_data, test_data = load_mockseq(torch.LongTensor, torch.Tensor)
        initial_embedding = torch.rand(5, 1536)

    # the first intensity is for the [MASK] token
    intensity_weights = 1 / torch.Tensor([1] + legend['avg_intensity'].tolist())

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    model = TimeSeriesTransformer(initial_embedding * intensity_weights.unsqueeze(1), n_heads=1, n_layers=3, dim_feedwordard=16)
    trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
    trainer.fit(model=model, train_dataloaders=train_loader)

if __name__ == '__main__':
    train_icat()