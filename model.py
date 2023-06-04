import lightning as L
import torch
import random
from positional_encodings.torch_encodings import PositionalEncoding1D

MASK_TOKEN = torch.LongTensor([0])

class TimeSeriesTransformer(L.LightningModule):
    def __init__(self, token_matrix,
                 n_heads=8,
                 n_layers=8, 
                 dropout=0.1, 
                 dim_feedwordard=2048, 
                 layer_norm_eps=0.00001) -> None:
        super().__init__()

        event_dim = token_matrix.shape[1]
        print(event_dim)
        print(token_matrix.shape)

        self.embed = torch.nn.Embedding.from_pretrained(token_matrix)
        encoder_layer = torch.nn.TransformerEncoderLayer(
                                            d_model=event_dim,
                                            nhead=n_heads, 
                                            dropout=dropout, 
                                            dim_feedforward=dim_feedwordard, 
                                            layer_norm_eps=layer_norm_eps)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer=encoder_layer, 
                                                       num_layers=n_layers)
        self.pos = PositionalEncoding1D(event_dim)

    def forward(self, masked_events):
        masked_events *= torch.sqrt(torch.tensor(self.embed.weight.shape[1]))
        masked_events += self.pos(masked_events)
        events = self.transformer(masked_events)
        return events
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def training_step(self, batch, batch_idx):
        events, intensities = batch
        intensities = intensities.clone()
        embeddings = self.embed(events)
        mask_idx = random.randint(0, len(intensities) - 1)
        intensities[mask_idx] = 1
        embeddings[mask_idx] = self.embed(MASK_TOKEN)
        embeddings *= intensities
        embeddings_pred = self(embeddings)
        loss = torch.nn.functional.mse_loss(embeddings_pred, events)
        self.log('train_loss', loss)
        return loss

    def unembed(self, token_vector):
        event = torch.cosine_similarity(self.embed.weight, token_vector, dim=1).argmax()
        intensity = token_vector.norm() / self.embed(event).norm()
        return event, intensity

    def predict_next(self, events, intensities):
        embeddings = self.embed(events)
        embeddings *= intensities
        embeddings = torch.vstack((embeddings, torch.zeros_like(embeddings[0])))
        embeddings_pred = self(events, intensities)
        return self.unembed(embeddings_pred[-1])