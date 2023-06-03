from typing import Any
import pytorch_lightning as pl
import torch
import random
from positional_encodings.torch_encodings import PositionalEncoding1D

class TimeSeriesTransformer(pl.LightningModule):
    def __init__(self, emb_matrix, mask_event=-1, *args, **kwargs) -> None:
        super().__init__()

        self.emb_matrix = emb_matrix
        self.mask_event = mask_event
        event_dim = emb_matrix.shape[1]
        self.transformer = torch.nn.TransformerEncoder(event_dim, *args, **kwargs)
        self.pos = PositionalEncoding1D(event_dim)

    def forward(self, masked_events):
        masked_events *= torch.sqrt(torch.tensor(self.emb_matrix.shape[1]))
        masked_events += self.pos(masked_events)
        events = self.transformer(masked_events)
        return events
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def training_step(self, batch, batch_idx):
        events, intensities = batch
        intensities = intensities.copy()
        embeddings = self.embed(events, intensities)
        mask_idx = random.randint(0, len(intensities) - 1)
        intensities[mask_idx] = 1
        embeddings[mask_idx] = self.emb_matrix[self.mask_event]
        embeddings *= intensities
        embeddings_pred = self(events, intensities)
        return torch.nn.functional.mse_loss(embeddings_pred, events)
    
    def embed(self, events):
        return self.emb_matrix[events]

    def unembed(self, embedding):
        event = torch.cosine_similarity(self.emb_matrix, embedding, dim=1).argmax()
        intensity = embedding.norm() / self.emb_matrix[event].norm()
        return event, intensity

    def predict_next(self, events, intensities):
        embeddings = self.embed(events)
        embeddings *= intensities
        embeddings = torch.vstack((embeddings, torch.zeros_like(embeddings[0])))
        embeddings_pred = self(events, intensities)
        return self.unembed(embeddings_pred[-1])