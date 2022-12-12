import torch
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

PAD_ID = 0

class AudioClassification_Dataset(Dataset):

    def __init__(self, data_dir):
        self.samples = [self.load_npy_file(file) for file  in data_dir.iterdir() if file.name.endswith("npy")]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    @staticmethod
    def load_npy_file(file):
        with open(file, 'rb') as f:
            array = np.load(f)
        array = torch.tensor(array).T.to(device)
        label = int(file.name[0])
        return (array,label)

def pad_collate(batch):
    samples, labels = zip(*batch)
    lengths = [sample.shape[0] for sample in samples]
    source_padded = pad_sequence(samples, batch_first=True, padding_value=PAD_ID).to(device)
    labels = torch.tensor(labels).to(device)
    lengths = torch.tensor(lengths).to(device)
    return source_padded, labels, lengths

def get_data_loader(data_dir,batch_size):
    dataset = AudioClassification_Dataset(data_dir)
    dataloader = DataLoader(dataset, batch_size = batch_size, collate_fn=pad_collate)
    return dataloader


if __name__ == "__main__":
    cur_dir = Path.cwd()
    data_dir = cur_dir.parent / "data"
    audio_dir = data_dir / "audio_data"
    project_dir = audio_dir / "2022-12-07"
    wake_word_array_dir = audio_dir / "wake_word_data_array"
    trn_array_dir = wake_word_array_dir / "trn_set"
    val_array_dir = wake_word_array_dir / "val_set"

    batch_size = 4
    val_set = AudioClassification_Dataset(val_array_dir)
    val_loader = DataLoader(val_set, batch_size = batch_size, collate_fn=pad_collate)
    batch = next(iter(val_loader))
    samples,labels,lengths = batch
    print(samples.shape,labels.shape,lengths.shape)


    # array,label = val_set[0]
    # print(type(array),type(label))
    # print(array.shape,label)
    #print([sample.shape for sample in samples])s
    #print(samples.shape, labels.shape, lengths.shape)
    #print(labels.shape,lengths.shape)
    
