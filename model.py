import lightning as L
import torch
import random
from positional_encodings.torch_encodings import PositionalEncoding1D
from eval import eval_event_pred, eval_intensity_pred

MASK_TOKEN = torch.LongTensor([0])

def mle(event_dist, intensities):
    """
    Most Likely Event
    Maximum Likelihood Estimation of the next event in a sequence.
    """

    top = event_dist.argmin(dim=-1)
    return top, intensities[:,top]

def embedding_transform(intensity_stds):
    def transform(embeddings):
        embeddings = torch.FloatTensor(embeddings)
        stds = torch.FloatTensor(intensity_stds)
        return embeddings / stds.unsqueeze(1)
    return transform

def patient_transform(avg_intensities):
    avg_intensities = torch.FloatTensor(avg_intensities)

    def transform(events, intensities):
        events = torch.LongTensor(events)
        intensities = torch.FloatTensor(intensities)
        avgs = avg_intensities[events]
        return events, intensities - avgs
    return transform

class TimeSeriesTransformer(L.LightningModule):
    def __init__(self, token_matrix,
                 n_heads=8,
                 n_layers=8, 
                 dropout=0.1, 
                 dim_feedwordard=2048, 
                 layer_norm_eps=0.00001) -> None:
        super().__init__()

        event_dim = token_matrix.shape[1]

        self.embed = torch.nn.Embedding.from_pretrained(token_matrix, 
                                                        freeze=False)
        encoder_layer = torch.nn.TransformerEncoderLayer(
                                            d_model=event_dim,
                                            nhead=n_heads, 
                                            dropout=dropout, 
                                            dim_feedforward=dim_feedwordard, 
                                            layer_norm_eps=layer_norm_eps)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer=encoder_layer, 
                                                       num_layers=n_layers)
        self.pos = PositionalEncoding1D(event_dim)
        self.loss_f = torch.nn.MSELoss()

    def forward(self, masked_events):
        masked_events *= torch.sqrt(torch.tensor(self.embed.weight.shape[1]))
        masked_events += self.pos(masked_events)
        events = self.transformer(masked_events)
        return events
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def unembed(self, token_vector):
        emb_norm = self.embed.weight / self.embed.weight.norm(dim=-1).unsqueeze(-1)
        proj_intensity = token_vector @ self.embed.weight.T
        event_dist = (token_vector.unsqueeze(-2) - proj_intensity.unsqueeze(-1) * emb_norm.unsqueeze(-3)).norm(dim=-1)
        return event_dist, proj_intensity

    def predict(self, events, intensities):
        embeddings = self.embed(events)
        embeddings *= intensities
        embeddings = torch.vstack((embeddings, self.embed(MASK_TOKEN)))
        embeddings_pred = self(events, intensities)
        return self.unembed(embeddings_pred[:,-1])
    
    def step(self, events, intensities, mode='train'):
        """
        A training or validation step

        events: batch_size x seq_len
        intensities: batch_size x seq_len
        mode: 'train' or 'val'
        """

        mask_idx = random.randint(0, intensities.shape[1] - 1)

        if mode == 'val':
            event_tail = events[:,mask_idx:]
            intensity_tail = intensities[:,mask_idx:]

            events = events[:,:mask_idx+1]
            intensities = intensities[:,:mask_idx+1]

        embeddings = self.embed(events) * intensities.unsqueeze(-1)
        masked_embeddings = embeddings.clone()
        masked_embeddings[:,mask_idx] = self.embed(MASK_TOKEN)
        embeddings_pred = self(masked_embeddings)

        if mode == 'val':
            event_dist, intensity_proj = self.unembed(embeddings_pred[:,-1])
            next_event, next_intensity = mle(event_dist, intensity_proj)
            event_eval_hard, event_eval_soft = eval_event_pred(event_tail.numpy(), next_event.numpy())
            self.log('event_eval_hard', event_eval_hard)
            self.log('event_eval_soft', event_eval_soft)
            in_eval = eval_intensity_pred(event_tail, intensity_tail, intensity_proj)
            self.log('intensity_eval', in_eval)

        return self.loss_f(embeddings_pred, embeddings)
    
    def training_step(self, batch, batch_idx):
        events, intensities = batch
        loss = self.step(events, intensities, mode='train')
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        events, intensities = batch
        loss = self.step(events, intensities, mode='val')
        self.log('val_loss', loss)