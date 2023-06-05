from model import TimeSeriesTransformer, embedding_transform, patient_transform
import torch
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers
import click

import data
import mock

N_HEADS = 3
N_LAYERS = 3
DIM = 15
DIM_FEEDFORWARD = 15

@click.command()
@click.option('--real-data/--mock-data', default=True)
@click.option('--accelerator', default='auto')
def train_icat(real_data, accelerator):
    if real_data:
        dataset = data
        from initemb import embed
    else:
        dataset = mock
        embed = lambda legend: torch.rand(len(legend['label']) + 1, DIM)

    legend = dataset.load_legend()

    # the first intensity is for the [MASK] token
    intensity_means = [1] + legend['intensity_mean'].tolist()
    intensity_stds = torch.Tensor([1] + legend['intensity_std'].tolist())

    transf = patient_transform(intensity_means)
    train_data = dataset.load_train(transf)
    test_data = dataset.load_test(transf)

    initial_embedding = torch.Tensor(embed(legend))
    initial_embedding = embedding_transform(intensity_stds)(initial_embedding)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    model = TimeSeriesTransformer(initial_embedding, 
                                  n_heads=N_HEADS, 
                                  n_layers=N_LAYERS, 
                                  dim_feedwordard=DIM_FEEDFORWARD)
    
    es = EarlyStopping(monitor='val_loss')

    trainer = L.Trainer(limit_train_batches=100, 
                        max_epochs=10, 
                        accelerator=accelerator,
                        logger=loggers.WandbLogger(project='icat'),
                        log_every_n_steps=1,
                        callbacks=[es],
                        precision=16)
    
    trainer.fit(model=model, 
                train_dataloaders=train_loader, 
                val_dataloaders=test_loader)

if __name__ == '__main__':
    train_icat()