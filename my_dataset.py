from datasets import get_dataset

def load_dataset(name, split='train', **kwargs):
    return get_dataset(name, split=split, **kwargs)