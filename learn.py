from model import TimeSeriesTransformer, PatientTransform
import torch
import lightning as L
from lightning.pytorch import loggers, callbacks
import click

import data
import mock

N_HEADS = 3
N_LAYERS = 3
DIM = 15
DIM_FEEDFORWARD = 15
LR = 1e-3
GRAD_CLIP_VAL = 0.25

@click.command()
@click.option('--real-data/--mock-data', default=True)
@click.option('--accelerator', default='auto')
@click.option('--slurm', is_flag=True)
def train_icat(real_data, accelerator, slurm):
    if real_data:
        dataset = data
        from initemb import embed
    else:
        dataset = mock
        embed = lambda legend, dim: torch.rand(len(legend['label']) + 1, dim)

    legend = dataset.load_legend()

    # the first intensity is for the [MASK] token
    intensity_means = [float('nan')] + legend['intensity_mean'].tolist()
    intensity_stds = torch.Tensor([float('nan')] + legend['intensity_std'].tolist())

    pt = PatientTransform(intensity_means, intensity_stds)
    train_data = dataset.load_train(pt)
    test_data = dataset.load_test(pt)

    initial_embedding = torch.Tensor(embed(legend, dim=DIM))

    train_loader = torch.utils.data.DataLoader(train_data, 
                                               batch_size=1, 
                                               shuffle=True, 
                                               num_workers=12, 
                                               prefetch_factor=256)
    test_loader = torch.utils.data.DataLoader(test_data, 
                                              batch_size=1, 
                                              shuffle=False, 
                                              num_workers=2, 
                                              prefetch_factor=16)

    model = TimeSeriesTransformer(initial_embedding, 
                                  n_heads=N_HEADS, 
                                  n_layers=N_LAYERS, 
                                  dim_feedwordard=DIM_FEEDFORWARD)
    
    trainer_opts = {
        'accelerator': accelerator,
        'log_every_n_steps': 1,
        'gradient_clip_val': GRAD_CLIP_VAL,
        'logger': loggers.WandbLogger(project='icat', log_model=True),
        'callbacks': [callbacks.ModelCheckpoint(monitor='val_loss',mode='min')]
    }

    if slurm:
        trainer_opts['devices'] = 2
        trainer_opts['num_nodes'] = 1
        trainer_opts['strategy'] = 'ddp'

    trainer = L.Trainer(**trainer_opts)
    
    trainer.fit(model=model, 
                train_dataloaders=train_loader, 
                val_dataloaders=test_loader,
                ckpt_path='latest')

if __name__ == '__main__':
    import lovely_tensors as lt
    lt.monkey_patch()
    train_icat()