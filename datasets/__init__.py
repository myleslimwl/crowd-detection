from datasets.shanghaitech import ShanghaiTechDataset
from datasets.mall import MallDataset
from datasets.jhu import JHUCrowdDataset

def get_dataset(name, split='train', **kwargs):
    if name.lower() == 'shanghaitech':
        return ShanghaiTechDataset(split=split, **kwargs)
    elif name.lower() == 'mall':
        return MallDataset(split=split, **kwargs)
    elif name.lower() == 'jhu':
        return JHUCrowdDataset(split=split, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {name}")