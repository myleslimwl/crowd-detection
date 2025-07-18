from datasets.shanghaitech import ShanghaiTechDataset
from datasets.jhu import JHUCrowdDataset
from datasets.mall import MallDataset

def load_dataset(name, split="train"):
    if name == "shanghaitech":
        return ShanghaiTechDataset(root="data/shanghaitech", split=split)
    elif name == "jhu":
        return JHUCrowdDataset(root="data/jhu", split=split)
    elif name == "mall":
        return MallDataset(root="data/mall", split=split)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

