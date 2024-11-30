import time
import torch
import torch.nn as nn
import sys
from math import ceil
from torch.cuda.amp import GradScaler, autocast
from utils import batchwise_sample
from dataset_utils import get_discriminator_dataloader_for_C, prepare_discriminator_data, get_discriminator_dataloader_for_GAN


def train_generator(vocab_size, model, epochs, dataloader, criterion, optimizer, device):
    model.train()
    scaler = GradScaler()  # Initialize GradScaler for mixed precision training
    epoch_times = []

    for epoch in range(epochs):
        start_time = time.time()  # Record start time of the epoch
        running_loss = torch.tensor(0.0, device=device)  # Accumulate loss on GPU

        for input_seq, target_seq in dataloader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            optimizer.zero_grad()
            with autocast():  # Enable mixed precision
                outputs = model(input_seq)
                target_seq = target_seq.contiguous().view(-1)
                outputs = outputs.view(-1, vocab_size)
                loss = criterion(outputs, target_seq)

            scaler.scale(loss).backward()  # Scale the loss and call backward
            scaler.step(optimizer)  # Step the optimizer
            scaler.update()  # Update the scale for next iteration

            running_loss += loss.detach()  # Accumulate loss directly on GPU

        end_time = time.time()  # Record end time of the epoch
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)

        epoch_loss = running_loss.item() / len(dataloader)  # Transfer to CPU once per epoch
        print(f"Epoch {epoch+1} loss: {epoch_loss:.3f}")

        if epoch == 0:  # Print time estimation after first epoch
            estimated_time_minutes = epoch_time / 60  # in minutes
            estimated_total_time_hours = (epoch_time * epochs) / 3600  # in hours
            print(f"Time for one epoch: {estimated_time_minutes:.2f} minutes")
            print(f"Estimated total training time: {estimated_total_time_hours:.2f} hours")

    total_time_hours = sum(epoch_times) / 3600  # Calculate total time in hours
    print(f"Total training time: {total_time_hours:.2f} hours")



def train_generatorPG(generator, discriminator, optimizer, device, len_of_GDL, epochs, batch_size, start_token, generate_length, temperature):
    generator.train()
    scaler = GradScaler()  # Initialize GradScaler for mixed precision training
    epoch_times = []

    for epoch in range(epochs):
        start_time = time.time()  # Record start time of the epoch
        running_loss = torch.tensor(0.0, device=device)  # Accumulate loss on GPU
        for _ in range(len_of_GDL):
            # Sample a batch of sequences from the generator
            _, generated_sequences = generator.sample_generator(start_token, batch_size, generate_length, temperature)

            # Construct input and target sequences
            input_seq = generated_sequences[:, :-1]  # All tokens except the last
            target_seq = generated_sequences[:, 1:]  # All tokens except the first

            input_seq, target_seq = input_seq.to(device), target_seq.to(device)

            # Generate rewards using the discriminator

            with torch.no_grad():
                rewards = discriminator.batchClassify(target_seq)

            optimizer.zero_grad()
            with autocast():  # Enable mixed precision
                pg_loss = generator.batchPGLossNEW(input_seq, target_seq, rewards)

            scaler.scale(pg_loss).backward()  # Scale the loss and call backward

            scaler.step(optimizer)  # Step the optimizer

            scaler.update()  # Update the scale for next iteration

            running_loss += pg_loss.detach()  # Accumulate loss directly on GPU
        end_time = time.time()  # Record end time of the epoch
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)

        epoch_loss = running_loss.item() / len_of_GDL  # Transfer to CPU once per epoch
        print(f"Epoch {epoch+1} policy gradient loss: {epoch_loss:.3f}")

        if epoch == 0:  # Print time estimation after first epoch
            estimated_time_minutes = epoch_time / 60  # in minutes
            estimated_total_time_hours = (epoch_time * epochs) / 3600  # in hours
            print(f"Time for one epoch: {estimated_time_minutes:.2f} minutes")
            print(f"Estimated total training time: {estimated_total_time_hours:.2f} hours")

    total_time_hours = sum(epoch_times) / 3600  # Calculate total time in hours
    print(f"Total training time: {total_time_hours:.2f} hours")


def train_discriminator(discriminator, dis_opt, pos_dataloader, POS_NEG_SAMPLES, generator, BATCH_SIZE=32, d_steps=1, epochs=1, start_token='<s>', generate_length=100):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """
    CUDA = torch.cuda.is_available()

    pos_val = []
    for input_seq, _ in pos_dataloader:
        pos_val.extend(input_seq.tolist())  # Convert tensors to lists and extend pos_val
        if len(pos_val) >= 100:
            pos_val = pos_val[:100]  # Ensure we only have 100 samples
            break

    neg_val_texts, neg_val = generator.sample_generator(start_token, batch_size=100, generate_length=100, temperature=1.0)
    val_loader = get_discriminator_dataloader_for_GAN(pos_val, neg_val.tolist(), batch_size=100, shuffle=False)

    # Extract val_inp and val_target from the DataLoader
    val_batch = next(iter(val_loader))
    val_inp, val_target = val_batch

    for d_step in range(d_steps):
        # Generate positive samples using the batchwise_sample function
        pos_samples_texts, s = batchwise_sample(generator, start_token, generate_length, POS_NEG_SAMPLES, BATCH_SIZE, temperature=1.0)
        real_data_samples = []
        for input_seq, _ in pos_dataloader:
            real_data_samples.extend(input_seq.tolist())
            if len(real_data_samples) >= POS_NEG_SAMPLES:
                real_data_samples = real_data_samples[:POS_NEG_SAMPLES]
                break
        real_data_samples = torch.tensor(real_data_samples, dtype=torch.long)

        print("Real samples size:", real_data_samples.size())
        print("Fake samples size:", s.size())

        dis_inp, dis_target = prepare_discriminator_data(real_data_samples, s, gpu=CUDA)
        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE):
                inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out > 0.5) == (target > 0.5)).data.item()

                if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            total_acc /= float(2 * POS_NEG_SAMPLES)

            val_pred = discriminator.batchClassify(val_inp)
            print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((val_pred > 0.5) == (val_target > 0.5)).data.item() / 200.))








