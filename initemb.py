import torch
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt

@retry(wait=wait_random_exponential(), stop=stop_after_attempt(6))
def embed_events(labels):
    data = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=labels
    )['data']

    return torch.Tensor([datum['embedding'] for datum in data])

def init_embedding(legend, batch_size=100):
    event_type_count = len(legend)
    max_event_id = legend['event_id'].max() 
    labels = legend['label'].tolist()

    try:
        initial_embedding = torch.load('initial_embedding.pt')
    except FileNotFoundError:
        initial_embedding = torch.Tensor(embed_events(['[MASK]']))

    del labels[:len(initial_embedding)-1]

    while labels:
        initial_embedding = torch.vstack((initial_embedding, embed_events(labels[:batch_size])))
        del labels[:batch_size]
        torch.save(initial_embedding, 'initial_embedding.pt')

    # Trying to prevent off-by-one errors (wish me luck)
    # initial_embedding[0] is the embedding for the [MASK] token
    # initial_embedding[event_id] is the embedding for the event with that id
    assert len(initial_embedding) == event_type_count + 1
    assert len(initial_embedding) == max_event_id

    return initial_embedding

if __name__ == '__main__':
    from data import load_mimicseq
    legend, train_data, test_data = load_mimicseq()
    init_embedding(legend)