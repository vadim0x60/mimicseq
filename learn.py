from model import TimeSeriesTransformer
from data import load_mimicseq
import torch
import pytorch_lightning as pl
from initemb import init_embedding

legend, train_data, test_data = load_mimicseq()
initial_embedding = init_embedding(legend)
intensity_weights = 1 / torch.Tensor(legend['intensity'].tolist())

train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
model = TimeSeriesTransformer(initial_embedding * intensity_weights, num_layers=4)
trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=model, train_dataloaders=train_loader)