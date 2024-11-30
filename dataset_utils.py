import os
import random
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


def prepare_dataset(file_path, sequence_length):
    """
    Prepares the dataset for model training.
    
    Args:
        file_path (str): Path to the text file containing the dataset.
        sequence_length (int): The length for sample sequence.
        
    Returns:
        samples (list of list of str): List of sequences of words.
        vocab_size (int): The size of the vocabulary.
        word_to_int (dict): Dictionary mapping words to their integer indices.
        int_to_word (dict): Dictionary mapping integer indices to words.
    """
    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Tokenize the text into words
    words = text.split()
    word_counts = Counter(words)

    # Create vocabulary
    vocab = list(word_counts.keys())
    start_token = '<s>'
    if start_token not in vocab:
        vocab.append(start_token)

    vocab_size = len(vocab)
    word_to_int = {word: i for i, word in enumerate(vocab)}
    int_to_word = {i: word for word, i in word_to_int.items()}

    # Create sequences
    samples = [words[i:i+sequence_length+1] for i in range(len(words) - sequence_length)]

    return samples, vocab_size, word_to_int, int_to_word



def split_samples(samples, test_size=0.5, random_state=42):
    """
    Splits the samples into generator samples and discriminator samples.

    Args:
        samples (list): The list of samples to be split.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.

    Returns:
        gen_samples (list): The generator samples.
        dis_samples (list): The discriminator samples.
    """
    gen_samples, dis_samples = train_test_split(samples, test_size=test_size, random_state=random_state)
    return gen_samples, dis_samples



def generate_fake_samples(real_samples, num_fakes=5, num_samples_to_select=None):
    """
    Generates fake samples from real samples.

    Args:
        real_samples (list of list of str): The list of real samples.
        num_fakes (int): Number of fake samples to generate for each real sample.
        num_samples_to_select (int): Number of samples to select from the generated fake samples.

    Returns:
        fake_samples (list of list of str): The list of generated fake samples.
    """
    fake_samples = []

    for sample in real_samples:
        words = sample[:]
        sample_length = len(words)
        for _ in range(num_fakes):
            # Generate a fake sample by shuffling words
            shuffled_words = words[:]
            random.shuffle(shuffled_words)
            fake_samples.append(shuffled_words)

            # Generate a fake sample by repeating random words
            repeated_words = [random.choice(words) for _ in range(sample_length)]
            fake_samples.append(repeated_words)

            # Generate a fake sample by mixing shuffled and repeated words
            mixed_words = [random.choice(words) if random.random() < 0.5 else word for word in words]
            random.shuffle(mixed_words)
            fake_samples.append(mixed_words)

    # Ensure the number of samples to select is less than or equal to the length of the list
    if num_samples_to_select is not None:
        if num_samples_to_select > len(fake_samples):
            raise ValueError("The number of samples to select exceeds the total number of available samples.")
        
        # Randomly select the specified number of samples
        fake_samples = random.sample(fake_samples, num_samples_to_select)

    return fake_samples



class DiscriminatorDataset(Dataset):
    def __init__(self, pos_samples, neg_samples, word_to_int):
        """
        Inputs:
            - pos_samples: List of lists containing positive samples (lists of words)
            - neg_samples: List of lists containing negative samples (lists of words)
            - word_to_int: Dictionary mapping words to integer indices
        """
        self.pos_samples = [torch.LongTensor([word_to_int[word] for word in sample]) for sample in pos_samples]
        self.neg_samples = [torch.LongTensor([word_to_int[word] for word in sample]) for sample in neg_samples]
        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)

    def __len__(self):
        return self.num_pos + self.num_neg

    def __getitem__(self, idx):
        if idx < self.num_pos:
            return self.pos_samples[idx], torch.tensor(1.0)  # 1.0 for real samples
        else:
            return self.neg_samples[idx - self.num_pos], torch.tensor(0.0)  # 0.0 for fake samples

def get_discriminator_dataloader_for_C(pos_samples, neg_samples, batch_size, word_to_int, shuffle=True):
    """
    Prepares the data for training the discriminator as a normal classifier.

    Inputs:
        - pos_samples: List of lists containing positive samples (lists of words)
        - neg_samples: List of lists containing negative samples (lists of words)
        - batch_size: Batch size
        - word_to_int: Dictionary mapping words to integer indices
        - shuffle: Whether to shuffle the data (default: True)

    Returns:
        - DataLoader instance
    """
    # Create dataset
    dataset = DiscriminatorDataset(pos_samples, neg_samples, word_to_int)

    # Create DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



class GeneratorDataset(Dataset):
    def __init__(self, samples, word_to_int, start_token='<s>'):
        """
        Inputs:
            - samples: List of lists containing samples (lists of words)
            - word_to_int: Dictionary mapping words to integer indices
            - start_token: Start token to prepend to the input sequence
        """
        self.samples = samples
        self.word_to_int = word_to_int
        self.start_token = start_token

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_seq = [self.start_token] + sample[:-1]
        target_seq = sample
        input_seq_tensor = torch.LongTensor([self.word_to_int[word] for word in input_seq])
        target_seq_tensor = torch.LongTensor([self.word_to_int[word] for word in target_seq])
        return input_seq_tensor, target_seq_tensor
    


def get_generator_dataloader(samples, word_to_int, batch_size, shuffle=True):
    """
    Prepares DataLoader for the TextDataset.

    Inputs:
        - samples: List of samples (lists of words)
        - word_to_int: Dictionary mapping words to integer indices
        - batch_size: Batch size
        - shuffle: Whether to shuffle the data (default: True)

    Returns:
        - DataLoader instance
    """
    dataset = GeneratorDataset(samples, word_to_int)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)




def prepare_discriminator_data(real_samples, generated_samples, gpu=False):
    """
    Prepares the discriminator input and target data.

    Inputs:
        - real_samples: Tensor of real samples
        - generated_samples: Tensor of generated samples
        - gpu: Boolean indicating whether to move tensors to GPU

    Returns:
        - dis_inp: Tensor containing concatenated real and generated samples
        - dis_target: Tensor containing labels (0 for fake, 1 for real)
    """
    # Create labels
    batch_size = real_samples.size(0)
    real_labels = torch.ones(batch_size, dtype=torch.float)
    fake_labels = torch.zeros(batch_size, dtype=torch.float)

    # Concatenate real and generated samples
    dis_inp = torch.cat((real_samples, generated_samples), 0)

    # Create target labels
    dis_target = torch.cat((real_labels, fake_labels), 0)

    # Ensure dis_target has shape (batch_size,)
    dis_target = dis_target.squeeze()  # Remove extra dimension if present

    if gpu:
        dis_inp = dis_inp.cuda()
        dis_target = dis_target.cuda()

    return dis_inp, dis_target

class DiscriminatorDatasetGAN(Dataset):
    def __init__(self, pos_samples, neg_samples):
        """
        Inputs:
            - pos_samples: List of lists containing positive samples (lists of word indices)
            - neg_samples: List of lists containing negative samples (lists of word indices)
        """
        self.pos_samples = [torch.LongTensor(sample) for sample in pos_samples]
        self.neg_samples = [torch.LongTensor(sample) for sample in neg_samples]
        self.num_pos = len(self.pos_samples)
        self.num_neg = len(self.neg_samples)

    def __len__(self):
        return self.num_pos + self.num_neg

    def __getitem__(self, idx):
        if idx < self.num_pos:
            return self.pos_samples[idx], torch.tensor(1.0)  # 1.0 for real samples
        else:
            return self.neg_samples[idx - self.num_pos], torch.tensor(0.0)  # 0.0 for fake samples

def get_discriminator_dataloader_for_GAN(pos_samples, neg_samples, batch_size, shuffle=True):
    """
    Prepares the data for training the discriminator as a normal classifier.

    Inputs:
        - pos_samples: List of lists containing positive samples (lists of word indices)
        - neg_samples: List of lists containing negative samples (lists of word indices)
        - batch_size: Batch size
        - shuffle: Whether to shuffle the data (default: True)

    Returns:
        - DataLoader instance
    """
    # Create dataset
    dataset = DiscriminatorDatasetGAN(pos_samples, neg_samples)

    # Create DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
