#!/usr/bin/env python3
from os import path
import argparse
import logging
import multiprocessing
import pickle
from collections import OrderedDict
import pdb
import time

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np

# Pytorch Imports
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from tqdm import tqdm

# Stencil imports
import embeddings
from dataset import SentimentDataset
import hyperparams

class SentimentNetwork(nn.Module):
    """ Sentiment classifier network. """

    def __init__(self, hidden_sz, embedding_lookup, rnn_layers=1, device=torch.device('cpu')):
        """
        The constructor for our net. The architecture of this net is the following:

            Embedding lookup -> RNN Encoder -> Linear layer -> Linear Layer

        You should apply a non-linear activation after the first linear layer
        and finally a softmax for the logits (optional if using CrossEntropyLoss). If you observe overfitting,
        dropout can be inserted between the two linear layers. Note that looking up the embeddings are handled
        by the modules in embeddings.py which can then be called in this net's forward pass.

        :param hidden_sz (int): The size of our RNN's hidden state
        :param embedding_lookup (str): The type of word embedding used.
                                       Either 'glove', 'elmo', 'both', or 'random'.
        :param num_layers (int): The number of RNN cells we want in our net (c.f num_layers param in torch.nn.LSTMCell)
        """
        super(SentimentNetwork, self).__init__()
        self.hidden_sz = hidden_sz
        self.rnn_layers = rnn_layers
        self.num_outputs = 2
        self.interm_size_1 = 256
        self.interm_size_2 = 64

        self.embedding_lookup = embedding_lookup.to(device) # instance of torch.nn.Module
        self.embed_size = embedding_lookup.embed_size # int value

        ## --- TODO: define the network architecture here ---
        ## Hint: you may wish to use nn.Sequential to stack linear layers, non-linear
        ##  activations, and dropout into one module
        ##
        ## Use GRU as your RNN architecture
        self.rnn_encoder = nn.GRU(self.embed_size, self.hidden_sz, num_layers=rnn_layers, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_sz, self.interm_size_1),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.interm_size_1, self.interm_size_2),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.interm_size_2, self.num_outputs)
        )

        self._dev = device
        self.to(device)

    def forward(self, tokens, seq_lens):
        """ TODO The forward pass for our model.
                 :param tokens: vectorized sequence inputs as token ids.
                 :param seq_lens: original sequence lengths of each input (prior to padding)
            You should return a tensor of (batch_size x 2) logits for each class per batch element.
        """
        curr_batch_size = seq_lens.shape[0] # hint: use this for reshaping RNN hidden state

        # 1. Grab the embeddings:
        embeds = self.embedding_lookup(tokens) # the embeddings for the token sequence

        # 2. Sort seq_lens and embeds in descending order of seq_lens. (check out torch.sort)
        #    This is expected by torch.nn.utils.pack_padded_sequence.
        sorted_seq_lens, perm_idx = seq_lens.sort(0, descending=True)
        sorted_seq_tensor = embeds[perm_idx]

        # 3. Obtain a PackedSequence object from pack_padded_sequence.
        #    Be sure to pass batch_first=True as the first dimension of our input is the batch dim.
        packed_input = pack_padded_sequence(sorted_seq_tensor, sorted_seq_lens, batch_first=True)

        # 4. Apply the RNN over the sequence of packed embeddings to obtain a sentence encoding.
        output, hidden = self.rnn_encoder(packed_input, self.init_hidden(curr_batch_size))

        # 5. Pass the sentence encoding (RNN hidden state) through your classifier net.
        logits = self.classifier(hidden).squeeze(0)

        # 6. Remember to unsort the output from step 5. If you sorted seq_lens and obtained a permutation
        #    over its indices (perm_ix), then the sorted indices over perm_ix will "unsort".
        #    For example:
        #       _, unperm_ix = perm_ix.sort(0)
        #       output = x[unperm_ix]
        #       return output
        _, unperm_idx = perm_idx.sort(0)
        unsorted_logits = logits[unperm_idx].squeeze(1)
        return unsorted_logits

    def init_hidden(self, batch_size):
        return torch.zeros(self.rnn_layers, batch_size, self.hidden_sz, device=self._dev)

def train(hp, embedding_lookup):
    """ This is the main training loop
            :param hp: HParams instance (see hyperparams.py)
            :param embedding_lookup: torch module for performing embedding lookups (see embeddings.py)
    """
    modes = ['train', 'dev']

    # Note: each of these are dicts that map mode -> object, depending on if we're using the training or dev data.
    datasets = {mode: SentimentDataset(args.data, mode) for mode in modes}
    data_sizes = {mode: len(datasets[mode]) for mode in modes} # hint: useful for averaging loss per batch
    dataloaders = {mode: DataLoader(datasets[mode], batch_size=hp.batch_size, shuffle=True, num_workers=6, drop_last=True) for mode in modes}

    model = SentimentNetwork(hp.rnn_hidden_size, embedding_lookup, device=DEV)
    print(model)
    loss_func = nn.CrossEntropyLoss() # TODO choose a loss criterion
    optimizer = optim.Adam(model.parameters(), lr=hp.learn_rate)

    train_loss = [] # training loss per epoch, averaged over batches
    dev_loss = [] # dev loss each epoch, averaged over batches

    # Note: similar to above, we can map mode -> list to append to the appropriate list
    losses = {'train': train_loss, 'dev': dev_loss}

    for epoch in range(1, hp.num_epochs+1):
        for mode in modes:
            running_loss = 0.0
            for (vectorized_seq, seq_len), label in tqdm(dataloaders[mode], desc='{}:{}/{}'.format(mode, epoch, hp.num_epochs)):
                vectorized_seq = vectorized_seq # note: we don't pass this to GPU yet
                seq_len = seq_len.to(DEV)
                label = label.long().to(DEV)
                if mode == 'train':
                    model.train() # tell pytorch to set the model to train mode
                    # TODO complete the training step. Hint: you did this for hw1
                    # don't forget to update running_loss as well
                    model.zero_grad()  # clear gradients (torch will accumulate them)
                    logits = model(vectorized_seq, seq_len)

                    # max_logits = torch.max(logits, dim=1)[0]
                    loss = loss_func(logits, label)

                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                else:
                    model.eval()
                    with torch.no_grad():
                        logits = model(vectorized_seq, seq_len)
                        loss = loss_func(logits, label)
                        running_loss += loss.item()
            avg_loss = running_loss/(data_sizes[mode]/64)
            losses[mode].append(avg_loss)
            print("{} Loss: {}".format(mode, avg_loss))

        torch.save(model.state_dict(), "{embed}_{i}_weights.pt".format(embed=args.embedding, i=epoch))

    # TODO plot train_loss and dev_loss
    plt.subplot(1, 2, 1)
    plt.plot(losses['train'])
    plt.title('Training loss')

    plt.subplot(1, 2, 2)
    plt.plot(losses['dev'])
    plt.title('Development loss')

    plt.savefig("{}_learning_curves_{}.png".format(args.embedding, time.time()))
    plt.close()

def evaluate(hp, embedding_lookup):
    """ This is used for the evaluation of the net. """
    mode = 'test' # use test data
    dataset = SentimentDataset(args.data, mode)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=6)
    model = SentimentNetwork(hp.rnn_hidden_size, embedding_lookup, device=DEV)
    model.load_state_dict(torch.load(args.restore))

    data_size = len(dataset)

    confusion = torch.zeros((2,2)) # TODO fill out this confusion matrix
    for (vectorized_seq, seq_len), label in tqdm(dataloader, ascii=True):
        vectorized_seq = vectorized_seq
        seq_len = seq_len.to(DEV)
        label = label.to(DEV)
        model.eval()
        with torch.no_grad():
            logits = model(vectorized_seq, seq_len)
            # TODO obtain a sentiment class prediction from output
            predicted_labels = torch.argmax(logits, dim=1)
            # assert sum(label.numpy().shape) == 1 and sum(predicted_labels.numpy().shape) == 1, "Expected one label"
            # confusion[label.numpy()[0]][predicted_labels.numpy()[0]] += 1
            confusion[label, predicted_labels] += 1

    accuracy = np.trace(confusion.numpy()) * 100. / data_size # TODO
    print("Sentiment Classification Accuracy of {:.2f}%".format(accuracy))
    print("Confusion matrix:")
    print(confusion)

def main():
    # Map word index back to the word's string. Due to a quirk in
    # pytorch's DataLoader implementation, we must produce batches of
    # integer id sequences. However, ELMo embeddings are character-level
    # and as such need the word. Additionally, we do not wish to restrict
    # ElMo to GloVe's vocabulary, and thus must map words to non-glove IDs here:
    with open(path.join(args.data, "idx2word.dict"), "rb") as f:
        idx2word = pickle.load(f)

    # --- Select hyperparameters and embedding lookup classes
    # ---  based on the embedding type:
    if args.embedding == "elmo":
        lookup = embeddings.Elmo(idx2word, device=DEV)
        hp = hyperparams.ElmoHParams()
    elif args.embedding == "glove":
        lookup = embeddings.Glove(args.data, idx2word, device=DEV)
        hp = hyperparams.GloveHParams()
    elif args.embedding == "both":
        lookup = embeddings.ElmoGlove(args.data, idx2word, device=DEV)
        hp = hyperparams.ElmoGloveHParams()
    elif args.embedding == "random":
        lookup = embeddings.RandEmbed(len(idx2word), device=DEV)
        hp = hyperparams.RandEmbedHParams(embed_size=lookup.embed_size)
    else:
        print("--embeddings must be one of: 'elmo', 'glove', 'both', or 'random'")

    # --- Either load and evaluate a trained model, or train and save a model ---
    if args.restore:
        evaluate(hp, lookup)
    else:
        train(hp, lookup)

if __name__ == '__main__':
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to data file", default="data")
    parser.add_argument("--embedding", type=str, help="embedding type")
    parser.add_argument("--device", type=str, help="cuda for gpu and cpu otherwise", default="cpu")
    parser.add_argument("--restore", help="filepath to restore")
    args = parser.parse_args()

    DEV = torch.device(args.device)

    print("######################### Using {} ############################".format(DEV))

    main()
