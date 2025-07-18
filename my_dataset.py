from datasets.shanghaitech import ShanghaiTechDataset

def load_dataset(name='shanghaitech', split='train'):
    """
    Load dataset based on name and split.
    
    Parameters:
        name (str): Dataset name ('shanghaitech' supported).
        split (str): 'train' or 'test'.

    Returns:
        Dataset object
    """
    name = name.lower()

    if name == 'shanghaitech':
        return ShanghaiTechDataset(split=split)
    else:
        raise ValueError(f"❌ Dataset '{name}' is not supported. Available: 'shanghaitech'")
