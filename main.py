import sys
import numpy as np
from typing import Tuple

import re
import time
import torch

from model import Attention, Encoder, Decoder, Seq2Seq

MAX_SEQUENCE_LENGTH = 29
TRAIN_URL = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt"


def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    print(true_expansion, " : ", pred_expansion, int(true_expansion == pred_expansion))
    return int(true_expansion == pred_expansion)


# --------- START OF IMPLEMENT THIS --------- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# device = torch.device("cpu")
alphabet_to_idx = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '+': 10,
                   '-': 11, '*': 12, '(': 13, ')': 14, 'x': 15, "$": 16, "&": 17, "@": 18, "<PAD>": 19, "<SOS>": 20,
                   "<EOS>": 21}

idx_to_alphabet = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '+',
                   11: '-', 12: '*', 13: '(', 14: ')', 15: 'x', 16: 'sin', 17: 'cos', 18: 'tan', 19: '<PAD>',
                   20: '<SOS>', 21: '<EOS>'}

INPUT_DIM = len(alphabet_to_idx)
OUTPUT_DIM = len(alphabet_to_idx)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 440
DEC_HID_DIM = 440
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
SRC_PAD_IDX = alphabet_to_idx["<PAD>"]

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, SRC_PAD_IDX, device)
model.load_state_dict(torch.load("models/polymodel-10-256-440-0.0.pt", map_location=torch.device(device)))
model = model.to(device)
model.eval()


def preprocess(sequence, word2id):
    sequence = re.sub('sin', '$', sequence)
    sequence = re.sub('cos', '&', sequence)
    sequence = re.sub('tan', '@', sequence)

    variable = re.sub('[^a-z]', '', sequence)
    assert len(set(variable)) == 1, f"Multiple Vars {variable}"

    sequence = re.sub('[a-z]', 'x', sequence)

    seq_tensor = []
    seq_tensor.append(word2id['<SOS>'])
    seq_tensor.extend([word2id[token] for token in sequence])
    seq_tensor.append(word2id['<EOS>'])
    seq_tensor = torch.Tensor(seq_tensor).long()
    seq_tensor = seq_tensor.unsqueeze(1).to(device)
    return seq_tensor, variable[0]


def predict(factors: str):
    model.eval()
    with torch.no_grad():
        inp, variable = preprocess(factors, alphabet_to_idx)
        inp_len = torch.tensor(len(inp)).unsqueeze(0).to(device)
        dummy_trg = torch.ones([31, 1], dtype=torch.int32).to(device) * alphabet_to_idx["<SOS>"]

        output_tensor = model(inp, inp_len, dummy_trg, 0)
        output_tensor = output_tensor.argmax(2)
        output_tensor = torch.transpose(output_tensor, 0, 1)
        for i in range(len(output_tensor)):
            result = ''.join([idx_to_alphabet[idx.item()] for idx in output_tensor[i][1:]])
            result = result.split("<EOS>")[0]
        result = result.replace("x", variable)
        return result


# --------- END OF IMPLEMENT THIS --------- #


def main(filepath: str):
    factors, expansions = load_file(filepath)
    t1 = time.time()
    pred = [predict(f) for f in factors[:1000]]
    scores = [score(te, pe) for te, pe in zip(expansions[:1000], pred)]
    t2 = time.time()
    print(f"Time Elapsed : {t2 - t1}")
    print(np.mean(scores))


if __name__ == "__main__":
    main("test.txt" if "-t" in sys.argv else "data/train.txt")