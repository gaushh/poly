import math
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim

from model import Attention, Encoder, Decoder, Seq2Seq
from data import get_loader


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def train(model, iterator, optimizer, criterion, clip):

    print_every = 1000
    print_running_loss = 0

    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = torch.transpose(batch[0], 0, 1).to(device)
        src_len = batch[1]
        trg = torch.transpose(batch[2], 0, 1).to(device)
        optimizer.zero_grad()
        output = model(src, src_len, trg)
        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        print_running_loss += loss.item()
        if (i + 1) % print_every == 0:
            print_loss_avg = print_running_loss / print_every
            print_running_loss = 0
            print("Iter: {} Train Loss: {:.3f}".format(i + 1, print_loss_avg))
            logging.info("Iter: {} Train Loss: {:.3f}".format(i + 1, print_loss_avg))
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src = torch.transpose(batch[0], 0, 1).to(device)
            src_len = batch[1]
            trg = torch.transpose(batch[2], 0, 1).to(device)
            output = model(src, src_len, trg, 0)  # turn off teacher forcing
            orig_output = output
            # orig_target = trg

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

            orig_output = orig_output.argmax(2)
            orig_output = torch.transpose(orig_output, 0, 1)

            for i in range(len(orig_output)):
                res = ''.join([idx_to_alphabet[idx.item()] for idx in orig_output[i][1:]])
                res = res.split("<EOS>")[0]

                src = ''.join([idx_to_alphabet[idx.item()] for idx in batch[2][i][1:]])
                src = src.split("<EOS>")[0]

                total += 1
                if src == res:
                    correct += 1

    val_accuracy = (correct * 100) / total
    return epoch_loss / len(iterator), val_accuracy


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":

    logging.basicConfig(filename="training_logs.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')

    logger = logging.getLogger()

    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alphabet_to_idx = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '+': 10,
                       '-': 11, '*': 12, '(': 13, ')': 14, 'x': 15, "$": 16, "&": 17, "@": 18, "<PAD>": 19, "<SOS>": 20,
                       "<EOS>": 21}

    idx_to_alphabet = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '+',
                       11: '-', 12: '*', 13: '(', 14: ')', 15: 'x', 16: '$', 17: '&', 18: '@', 19: '<PAD>', 20: '<SOS>',
                       21: '<EOS>'}

    dataset_path = "data/train.txt"
    train_dataloader, valid_dataloader = get_loader(dataset_path, alphabet_to_idx, batch_size=64, train_valid_ratio=0.85)

    INPUT_DIM = len(alphabet_to_idx)
    OUTPUT_DIM = len(alphabet_to_idx)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 440
    DEC_HID_DIM = 440
    ENC_DROPOUT = 0.0
    DEC_DROPOUT = 0.0
    SRC_PAD_IDX = alphabet_to_idx["<PAD>"]

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, device)

    model = model.to(device)

    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters())

    TRG_PAD_IDX = SRC_PAD_IDX
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    N_EPOCHS = 10
    CLIP = 1

    logging.info(f"ENC_EMB_DIM = {ENC_EMB_DIM}")
    logging.info(f"DEC_EMB_DIM = {DEC_EMB_DIM}")
    logging.info(f"ENC_HID_DIM = {ENC_HID_DIM}")
    logging.info(f"DEC_HID_DIM = {DEC_HID_DIM}")
    logging.info(f"ENC_DROPOUT = {ENC_DROPOUT}")
    logging.info(f"DEC_DROPOUT = {DEC_DROPOUT}")
    logging.info(f"N_EPOCHS = {N_EPOCHS}")

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
        valid_loss, val_accuracy = evaluate(model, valid_dataloader, criterion)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'models/polymodel-{N_EPOCHS}-{ENC_EMB_DIM}-{ENC_HID_DIM}-{ENC_DROPOUT}.pt')

        logging.info(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        logging.info('\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        logging.info(f'\t Val. Loss: {valid_loss:.3f} |  Val Acc: {val_accuracy}')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val Acc: {val_accuracy}')