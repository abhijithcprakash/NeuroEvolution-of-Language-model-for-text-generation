import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from math import ceil




def train_discriminator_C(discriminator, dis_dataloader, num_epochs=100, learning_rate=0.001,
                         gpu=False):
    
    if gpu:
        discriminator.cuda()

    # Define optimizer
    optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (seq_batch, target_batch) in enumerate(dis_dataloader):
            if gpu:
                seq_batch, target_batch = seq_batch.cuda(), target_batch.cuda()

            optimizer.zero_grad()
            loss = discriminator.batchBCELoss(seq_batch, target_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dis_dataloader)}")



def test_discriminator_C(discriminator, test_samples, word_to_int, gpu=False):
    """
    Classifies test samples using the trained discriminator model.

    Inputs:
        - discriminator: Trained Discriminator model
        - test_samples: List of lists containing test samples (lists of words)
        - word_to_int: Dictionary mapping words to integer indices
        - gpu: Boolean indicating whether to use GPU

    Returns:
        - List of classification scores
    """
    test_tensors = [torch.LongTensor([word_to_int[word] for word in sample]) for sample in test_samples]
    test_dataset = torch.stack(test_tensors)

    if gpu:
        test_dataset = test_dataset.cuda()

    with torch.no_grad():
        hidden = discriminator.init_hidden(len(test_dataset))
        scores = discriminator.batchClassify(test_dataset)

    return scores.cpu().numpy().tolist()



def return_int_vector(text, word_to_int, SEQUENCE_LENGTH):
    words = text.split()
    input_seq = torch.LongTensor([word_to_int[word] for word in words[-SEQUENCE_LENGTH:]]).unsqueeze(0)
    return input_seq

def sample_next(predictions, temperature=1.0):
    """
    Sampling with temperature.
    """
    # Apply softmax to get probabilities.
    probabilities = F.softmax(predictions[:, -1, :] / temperature, dim=-1).cpu().numpy()

    # Sample from the distribution.
    next_token = np.random.choice(len(probabilities[0]), p=probabilities[0])

    return next_token

def text_generator(gen, device, sentence, int_to_word, word_to_int, generate_length, SEQUENCE_LENGTH, temperature=1.0):
    gen.eval()
    sample = sentence
    for i in range(generate_length):
        int_vector = return_int_vector(sample, word_to_int, SEQUENCE_LENGTH)
        input_tensor = int_vector.to(device)
        with torch.no_grad():
            predictions = gen(input_tensor)
        next_token = sample_next(predictions, temperature)
        next_word = int_to_word[next_token]

        # Add the next word to the sample
        sample += ' ' + next_word

        # If the next word ends a sentence, add a newline
        if next_word in {'.', '!', '?'}:
            sample += '\n'

    # Ensure the output ends with a newline character
    if not sample.endswith('\n'):
        sample += '\n'

    print(sample)





def batchwise_sample(gen, start_token, generate_length, num_samples, batch_size, temperature=1.0):
    """
    Sample num_samples samples batch_size samples at a time from gen using sample_generator method.
    Does not require GPU since gen.sample() takes care of that.

    Inputs:
        - gen: The generator instance
        - start_token: The start token for the generator
        - generate_length: The length of sequences to generate
        - num_samples: Total number of samples to generate
        - batch_size: Number of samples to generate per batch
        - temperature: Sampling temperature

    Returns:
        - A tuple containing:
          1. List of generated samples as text
          2. Tensor containing num_samples generated sequences
    """
    samples_texts = []
    samples = []
    for _ in range(int(ceil(num_samples / float(batch_size)))):
        batch_texts, batch_samples = gen.sample_generator(start_token, batch_size, generate_length, temperature)
        samples_texts.extend(batch_texts)
        samples.append(batch_samples)

    return samples_texts[:num_samples], torch.cat(samples, 0)[:num_samples]
