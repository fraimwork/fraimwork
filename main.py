import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tokenization and preprocessing
def tokenize(text):
    text = text.lower().strip()
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text = text.strip()
    return text.split()

# Build vocabulary and map words to indices
def build_vocab(sentences, max_vocab_size=10000):
    all_words = [word for sentence in sentences for word in tokenize(sentence)]
    vocab = Counter(all_words)
    vocab = vocab.most_common(max_vocab_size - 2)
    vocab = ['<PAD>', '<UNK>'] + [word for word, _ in vocab]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

# Convert sentences to indices
def sentence_to_indices(sentence, word2idx):
    return [word2idx.get(word, word2idx['<UNK>']) for word in tokenize(sentence)]

# Padding sequences to the same length
def pad_sequences(sequences, max_len):
    return [seq + [0] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in sequences]

# Define a custom dataset
class TranslationDataset(Dataset):
    def __init__(self, source_sentences, target_sentences, source_word2idx, target_word2idx, max_len=20):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.source_word2idx = source_word2idx
        self.target_word2idx = target_word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        source_seq = sentence_to_indices(self.source_sentences[idx], self.source_word2idx)
        target_seq = sentence_to_indices(self.target_sentences[idx], self.target_word2idx)
        source_seq = pad_sequences([source_seq], self.max_len)[0]
        target_seq = pad_sequences([target_seq], self.max_len)[0]
        return torch.tensor(source_seq), torch.tensor(target_seq)
    
import pandas as pd

df = pd.read_csv('data/data.csv')

# Load data
english_sentences = df['english'].values
spanish_sentences = df['spanish'].values

print(f"Number of English sentences: {len(english_sentences)}")
print(f"Number of Spanish sentences: {len(spanish_sentences)}")

# Build vocabularies
source_word2idx, source_idx2word = build_vocab(english_sentences)
target_word2idx, target_idx2word = build_vocab(spanish_sentences)

# Split data into training and testing sets
train_source, test_source, train_target, test_target = train_test_split(english_sentences, spanish_sentences, test_size=0.2)

# Create datasets and dataloaders
train_dataset = TranslationDataset(train_source, train_target, source_word2idx, target_word2idx)
test_dataset = TranslationDataset(test_source, test_target, source_word2idx, target_word2idx)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)  # (batch_size, src_len, hid_dim)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs.permute(1, 0, 2)), dim=2)))
        attention = self.v(energy).squeeze(2)
        return nn.functional.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM((hid_dim * 2) + emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs.permute(1, 0, 2))
        rnn_input = torch.cat((embedded, weighted.permute(1, 0, 2)), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        output = torch.cat((embedded.squeeze(0), output.squeeze(0), weighted.squeeze(1)), dim=1)
        prediction = self.fc_out(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

# Training hyperparameters
INPUT_DIM = len(source_word2idx)
OUTPUT_DIM = len(target_word2idx)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
LEARNING_RATE = 0.001
N_EPOCHS = 10

# Initialize the encoder, decoder, and seq2seq model
attn = Attention(HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)
model = Seq2Seq(enc, dec, device).to(device)

# Initialize optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=0)


# Training loop
def train(model, loader, optimizer, criterion, clip=1):
    model.train()
    epoch_loss = 0
    for src, trg in loader:
        src, trg = src.transpose(0, 1).to(device), trg.transpose(0, 1).to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

# Train the model
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.3f}')

def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=20):
    model.eval()
    tokens = sentence_to_indices(sentence, src_vocab)
    tokens = torch.tensor(tokens).unsqueeze(1).to(device)
    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder(tokens)
    trg_indices = [trg_vocab['<sos>']]
    for _ in range(max_len):
        trg_tensor = torch.tensor([trg_indices[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell, encoder_outputs)
        pred_token = output.argmax(1).item()
        trg_indices.append(pred_token)
        if pred_token == trg_vocab['<eos>']:
            break
    trg_tokens = [target_idx2word[idx] for idx in trg_indices]
    return trg_tokens[1:]

# Test the model with some sentences
test_sentences = ["This is a test.", "How are you?", "Translate this sentence."]
for sentence in test_sentences:
    translation = translate_sentence(sentence, source_word2idx, target_word2idx, model, device)
    print(f'English: {sentence}')
    print(f'Spanish: {" ".join(translation)}')