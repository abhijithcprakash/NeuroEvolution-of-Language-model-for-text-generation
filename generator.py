import torch
import math
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



def generate_square_subsequent_mask(sz):
    """
    Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):
        """
        :param max_len: Input length sequence.
        :param d_model: Embedding dimension.
        :param dropout: Dropout value (default=0.1)
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Inputs of forward function
        :param x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[:, :x.size(1)]#self.pe[:x.size(0), :]
        return self.dropout(x)
    
class Generator(nn.Module):
    def __init__(self, word_to_int, int_to_word, sequence_length, device, vocab_size, embed_dim, num_layers, num_heads):
        super(Generator, self).__init__()
        self.word_to_int = word_to_int
        self.int_to_word = int_to_word
        self.device = device
        self.SEQUENCE_LENGTH = sequence_length
        self.vocab_size = vocab_size
        self.pos_encoder = PositionalEncoding(max_len=sequence_length+1, d_model=embed_dim)
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=num_layers,
        )
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)

    # Positional encoding is required. Else the model does not learn.
    def forward(self, x):
        emb = self.emb(x)
        # Generate input sequence mask with shape (SEQUENCE_LENGTH, SEQUENCE_LENGTH)
        input_mask = generate_square_subsequent_mask(x.size(1)).to(x.device)

        x = self.pos_encoder(emb)
        x = self.decoder(x, memory=x, tgt_mask=input_mask, memory_mask=input_mask)
        x = self.dropout(x)
        out = self.linear(x)
        return out

    # Next the functions are used to sample a batch of sentences form the model using <s>
    def sample_next_batch(self, predictions, temperature=1.0):
        """
        Sampling with temperature for a batch of sequences.
        """
        # Apply softmax to get probabilities
        probabilities = F.softmax(predictions[:, -1, :] / temperature, dim=-1).cpu().numpy()

        # Sample from the distribution for each sequence in the batch
        next_tokens = [np.random.choice(len(probabilities[0]), p=probabilities[i]) for i in range(probabilities.shape[0])]
        return torch.tensor(next_tokens, dtype=torch.long)

    def sample_generator(self, start_token, batch_size, generate_length, temperature=1.0):
        self.eval()
        start_token_id = self.word_to_int[start_token]  # Assuming you have a word_to_int dictionary
        generated_sequences = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=self.device)

        for _ in range(generate_length):
            with torch.no_grad():
                predictions = self.forward(generated_sequences)
            next_tokens = self.sample_next_batch(predictions, temperature).to(self.device)  # Ensure next_tokens is on the same device
            next_tokens = next_tokens.unsqueeze(1)
            generated_sequences = torch.cat((generated_sequences, next_tokens), dim=1)

            # Stop if all sequences in the batch have reached SEQUENCE_LENGTH
            if generated_sequences.size(1) >= self.SEQUENCE_LENGTH + 1:
                break

        # Convert generated sequences from IDs to words
        generated_texts = []
        for seq in generated_sequences:
            words = [self.int_to_word[token.item()] for token in seq]
            generated_texts.append(' '.join(words))
        generated_sequences = generated_sequences.long()

        return generated_texts, generated_sequences

    def batchCELoss(self, inp, target):
        """
        Returns the CrossEntropy Loss for predicting target sequence.

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len

            inp should be target with <s> (start letter) prepended
        """
        loss_fn = nn.CrossEntropyLoss()
        batch_size, seq_len = inp.size()
        predictions = self.forward(inp)  # batch_size x seq_len x vocab_size
        predictions = predictions.view(-1, self.vocab_size)
        target = target.contiguous().view(-1)  # Flatten target
        loss = loss_fn(predictions, target)
        return loss

    


    def batchPGLossNEW(self, inp, target, reward):
      """
      Returns a loss that gives corresponding policy gradients .
      Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

      Inputs: inp, target, reward
          - inp: batch_size x seq_len
          - target: batch_size x seq_len
          - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                    sentence)

          inp should be target with <s> (start letter) prepended
      """

      batch_size, seq_len = inp.size()
      predictions = self.forward(inp)  # predictions: [batch_size, seq_len, vocab_size]
      log_probs = F.log_softmax(predictions, dim=-1)  # Compute log-softmax once

      # Ensure reward is on the same device as log_probs
      reward = reward.to(log_probs.device)

      # Reshape log_probs and target to gather log probabilities
      log_probs = log_probs.reshape(-1, log_probs.size(-1))  # [batch_size * seq_len, vocab_size]
      target = target.reshape(-1, 1)  # [batch_size * seq_len, 1]

      # Gather log probabilities corresponding to target tokens
      gathered_log_probs = log_probs.gather(1, target).reshape(batch_size, seq_len)  # [batch_size, seq_len]

      # Calculate the loss by weighting log probabilities with rewards
      loss = -gathered_log_probs * reward.unsqueeze(1)  # reward: [batch_size, 1]
      loss = loss.mean()  # Average over the batch

      return loss