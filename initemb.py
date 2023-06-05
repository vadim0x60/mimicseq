import numpy as np
import openai
from sklearn.decomposition import PCA
from tenacity import retry, wait_random_exponential, stop_after_attempt

@retry(wait=wait_random_exponential(), stop=stop_after_attempt(6))
def embed_events(labels):
    data = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=labels
    )['data']

    return np.array([datum['embedding'] for datum in data])

def openai_embed(legend, batch_size=100):
    event_type_count = len(legend)
    max_event_id = legend['event_id'].max() 
    labels = legend['label'].tolist()

    try:
        embedding = np.load('initial_embedding.npy')
    except FileNotFoundError:
        embedding = embed_events(['[MASK]'])

    del labels[:len(embedding)-1]

    while labels:
        embedding = np.vstack((embedding, embed_events(labels[:batch_size])))
        del labels[:batch_size]
        np.save('initial_embedding.npy', embedding)

    # Trying to prevent off-by-one errors (wish me luck)
    # embedding[0] is the embedding for the [MASK] token
    # embedding[event_id] is the embedding for the event with that id
    assert len(embedding) == event_type_count + 1
    assert len(embedding) == max_event_id

    return embedding

def embed(legend, dim=16, batch_size=100):
    try:
        embedding = np.load('embedding.npy')
        assert embedding.shape[0] == len(legend) + 1
        assert embedding.shape[1] == dim
    except (FileNotFoundError, AssertionError):
        initial_embedding = openai_embed(legend, batch_size)
        pca = PCA(n_components=dim)
        embedding = pca.fit_transform(initial_embedding)
        np.save('embedding.npy', embedding)
        return embedding

if __name__ == '__main__':
    from data import load_mimicseq
    legend, train_data, test_data = load_mimicseq()
    embed(legend)