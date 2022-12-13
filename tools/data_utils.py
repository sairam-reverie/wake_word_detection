import torch
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torchsampler import ImbalancedDatasetSampler


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
        name = file.name
        label = int(name[0])
        return (array,label,name)

def get_label(dataset):
    return [example[1] for example in dataset]

def pad_collate(batch):
    samples,labels,names = zip(*batch)
    lengths = [sample.shape[0] for sample in samples]
    source_padded = pad_sequence(samples, batch_first=True, padding_value=PAD_ID).to(device)
    labels = torch.tensor(labels).to(device)
    lengths = torch.tensor(lengths).to(device)
    return source_padded,labels,lengths,names

def get_data_loader(data_dir,batch_size,balanced=False):
    dataset = AudioClassification_Dataset(data_dir)
    if balanced:
        sampler=ImbalancedDatasetSampler(dataset,callback_get_label=get_label)
        dataloader = DataLoader(dataset, batch_size = batch_size, collate_fn=pad_collate,sampler=sampler)
    else:
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

    batch_size = 16
    val_set = AudioClassification_Dataset(val_array_dir)
    val_len = len(val_set)
    print(val_len)
    batch_size=10
    val_loader = get_data_loader(val_array_dir,batch_size,balanced=True)
    num_iterations = (val_len*2)//batch_size
    val_iter = iter(val_loader)
    for _ in range(num_iterations):
        batch = next(val_iter)
        print(batch[1])
    # #val_loader = DataLoader(val_set, batch_size = batch_size, collate_fn=pad_collate)
    # #labels = 
    # sampler=ImbalancedDatasetSampler(val_set,callback_get_label=get_label)
    # val_loader = DataLoader(dataset=val_set,sampler=sampler,batch_size=batch_size,collate_fn=pad_collate)
    # batch = next(iter(val_loader))
    # samples,labels,lengths,names = batch
    # print(samples.shape)
    # for i
    # labels = labels.tolist()
    # lengths = lengths.tolist()
    # for bunch in zip(labels,lengths,names):
    #     print(bunch)


    # array,label = val_set[0]
    # print(type(array),type(label))
    # print(array.shape,label)
    #print([sample.shape for sample in samples])s
    #print(samples.shape, labels.shape, lengths.shape)
    #print(labels.shape,lengths.shape)
    
