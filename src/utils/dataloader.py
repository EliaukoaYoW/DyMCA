import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader

def create_dataloder(input_ids, attention_mask, stance):
    dataset = TensorDataset(input_ids, attention_mask, stance)
    dataloder = DataLoader(dataset, shuffle=False, batch_size=64)
    return dataloder

def combine_datasets(orig_data, aug_data, aug_ratio):
    combined_data = orig_data.copy()
    aug_sample = aug_data.sample(frac=aug_ratio, random_state=56)
    combined_data = pd.concat([combined_data, aug_sample])
    combined_data = shuffle(combined_data, random_state=56)
    return combined_data
