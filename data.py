import re
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset


def load_file(file_path: str):
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, file_path, word2id):
        """Reads source and target sequences from txt files."""
        self.src_seqs, self.trg_seqs = load_file(file_path)
        self.num_total_seqs = len(self.src_seqs)
        self.word2id = word2id

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        src_seq = self.preprocess(src_seq, self.word2id)
        trg_seq = self.preprocess(trg_seq, self.word2id)
        return src_seq, trg_seq

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id):
        """Converts words to ids."""
        sequence = re.sub('sin', '$', sequence)
        sequence = re.sub('cos', '&', sequence)
        sequence = re.sub('tan', '@', sequence)
        sequence = re.sub('[a-z]', 'x', sequence)
        res = []
        res.append(word2id['<SOS>'])
        res.extend([word2id[token] for token in sequence])
        res.append(word2id['<EOS>'])
        res = torch.Tensor(res)
        return res


def collate_fn(data):

    def merge(sequences):
        lengths = torch.tensor([len(seq) for seq in sequences])
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seqs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)
    return src_seqs, src_lengths, trg_seqs, trg_lengths


def get_loader(file_path, word2id, batch_size=32, train_valid_ratio=0.8):

    dataset = Dataset(file_path, word2id)

    train_size = int(train_valid_ratio * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, (train_size, valid_size))
    # this will return (src_seqs, src_lengths, trg_seqs, trg_lengths) for each iteration

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=collate_fn)

    valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
    return train_dataloader, valid_dataloader