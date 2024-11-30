from dataset_utils import prepare_dataset, split_samples, generate_fake_samples
from dataset_utils import get_discriminator_dataloader_for_C, get_generator_dataloader
from generator import Generator
from discriminator import Discriminator
from utils import test_discriminator_C, train_discriminator_C
from utils import text_generator
from train import train_generator, train_generatorPG, train_discriminator
import torch.optim as optim
import torch
import sys
import torch.nn as nn



# Define the path to your dataset and the sequence length
file_path = "C:\\Users\\abhij\\CODES\\LLM and RL\\My_Coding\\Final Phase 2 Project\\captions_only.txt"

sequence_length = 100

# Call the function to prepare the dataset
samples, vocab_size, word_to_int, int_to_word = prepare_dataset(file_path, sequence_length)

print('Vocabulary Size: ', vocab_size)
print('Word to Index Mapping done ')  #, word_to_int)

gen_samples, dis_samples = split_samples(samples, test_size=0.5, random_state=42)

print('Generator Samples: ', len(gen_samples))
print('Discriminator Samples: ', len(dis_samples))

# Generate fake samples from the discriminator samples
num_samples_to_select = 238348
fake_samples = generate_fake_samples(dis_samples, num_fakes=1, num_samples_to_select=num_samples_to_select)

print('Number of Fake Samples Generated: ', len(fake_samples))

# Prepare data for training the discriminator as a normal classifier
BATCH_SIZE = 32
shuffle = True
pos_samples = dis_samples[:500]
neg_samples = fake_samples[:500]

# Create Discriminator DataLoader
dis_classifier_dataloader = get_discriminator_dataloader_for_C(pos_samples, neg_samples, BATCH_SIZE, word_to_int, shuffle)

print('DataLoader ready for training discriminator ')

# # Example of using the dataloader
# for batch_idx, (input_batch, target_batch) in enumerate(dis_classifier_dataloader):
#     print(f"Discriminator dataloader Batch {batch_idx + 1}: ")
#     print("Sequence Batch Shape: ", input_batch.shape)
#     print("Target Batch Shape: ", target_batch.shape)
#     break 


# Create Generator Dataloader
gen_samples = gen_samples[:500]

gen_dataloader = get_generator_dataloader(gen_samples, word_to_int, BATCH_SIZE, shuffle=True)

print('DataLoader ready for training generator ')


# # Example of using the dataloader
# for batch_idx, (input_seq, target_seq) in enumerate(gen_dataloader):
#     print(f"Generator dataloader Batch {batch_idx + 1}: ")
#     print("Input sequences shape: ", input_seq.shape)
#     print("Target sequences shape: ", target_seq.shape)
#     break  

# Initialize the Generator model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gen_model = Generator(word_to_int, int_to_word, sequence_length, device, vocab_size=vocab_size, 
                embed_dim=128, num_layers=2, num_heads=4).to(device)
print('Device is set and Generator Model Intialized ')

# Example usage of model
start_token = '<s>'
generated_texts, generated_sequences = gen_model.sample_generator(start_token, batch_size=5, generate_length=100, temperature=1.0)
# print('gen.sample size',generated_sequences.shape)
# # Print generated texts
# for text in generated_texts:
#     print(text)
print('Sentences sampled from the Generator ')


# Initialize the Discriminator model
embedding_dim = 128
hidden_dim = 64
dropout = 0.2
gpu = torch.cuda.is_available()

# Initialize the Discriminator
dis_model = Discriminator(embedding_dim, hidden_dim, vocab_size, max_seq_len=sequence_length, gpu=gpu, dropout=dropout)
if gpu:
    dis_model.cuda()
print('Discriminator model initialized')

# Testing the Discriminator working
# Example test samples (real and fake)
test_real_samples = dis_samples[:3]  # First 10 real samples
test_fake_samples = fake_samples[:3]  # First 10 fake samples

print('Testing the Discriminator')
# Get classification scores for real and fake samples
real_scores = test_discriminator_C(dis_model, test_real_samples, word_to_int, gpu=gpu)
fake_scores = test_discriminator_C(dis_model, test_fake_samples, word_to_int, gpu=gpu)

# Print results
print("Real sample scores:", real_scores)
print("Fake sample scores:", fake_scores)


# # # Training Discriminator
# # train_discriminator_C(dis_model, dis_classifier_dataloader, num_epochs=5, learning_rate=0.001,
# #                          gpu=gpu)

# print('Testing the Discriminator after training')
# # Get classification scores for real and fake samples
# real_scores = test_discriminator_C(dis_model, test_real_samples, word_to_int, gpu=gpu)
# fake_scores = test_discriminator_C(dis_model, test_fake_samples, word_to_int, gpu=gpu)

# # Print results
# print("Real sample scores:", real_scores)
# print("Fake sample scores:", fake_scores)


# Training the generator
print('Pre training Generator')
epochs = 30
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(gen_model.parameters(), lr=learning_rate)


train_generator(vocab_size, gen_model, epochs, gen_dataloader, criterion, optimizer, device)

# print('Lets test the generation of generator')

# sentences = [
#     "A"
# ]

# generate_length = 500
# for sentence in sentences:
#     print(f"PROMPT: {sentence}")
#     text_generator(gen_model, device, sentence, int_to_word, word_to_int, generate_length, SEQUENCE_LENGTH=sequence_length, temperature=1.0)

# print('Finished Generation')

print('checking Policy Gradient training of generator')
len_of_GDL = len(gen_dataloader)

train_generatorPG(gen_model, dis_model, optimizer, device, len_of_GDL, epochs=1, batch_size=BATCH_SIZE, 
                  start_token='<s>', generate_length=100, temperature=1.0)



pos_samples_for_dis = gen_samples[-500:]
pos_dataloader = get_generator_dataloader(pos_samples_for_dis, word_to_int, BATCH_SIZE, shuffle=True)
dis_opt = optim.Adam(dis_model.parameters(), lr=learning_rate)
POS_NEG_SAMPLES = 200


print('Pre training Discriminator')
train_discriminator(dis_model, dis_opt, pos_dataloader, POS_NEG_SAMPLES,
                     gen_model, BATCH_SIZE=32, d_steps=1, epochs=10, start_token='<s>', generate_length=100)


ADV_TRAIN_EPOCHS = 5
for epoch in range(ADV_TRAIN_EPOCHS):
    print('\n--------\nEPOCH %d\n--------' % (epoch+1))
    # TRAIN GENERATOR
    print('\nAdversarial Training Generator : ', end='')
    sys.stdout.flush()
    train_generatorPG(gen_model, dis_model, optimizer, device, len_of_GDL, epochs=3, batch_size=BATCH_SIZE, 
                  start_token='<s>', generate_length=100, temperature=1.0)

    # TRAIN DISCRIMINATOR
    print('\nAdversarial Training Discriminator : ')
    train_discriminator(dis_model, dis_opt, pos_dataloader, POS_NEG_SAMPLES,
                     gen_model, BATCH_SIZE=32, d_steps=1, epochs=10, start_token='<s>', generate_length=100)