import numpy as np
import pandas as pd
from pandas import DataFrame
from collections import namedtuple
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GRU, LSTM, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow import data as tfdata
from tensorflow import config as tfconfig
from tensorflow import function, GradientTape, sqrt, ones_like, convert_to_tensor, float32, nn,  reduce_mean, zeros_like
from tqdm import tqdm
import tensorflow as tf
from dataclasses import dataclass

import os
import sys
# Define the path one level up
parent_directory = os.path.join(os.getcwd(), '../../Workspace/Users/iaaph@energinet.dk/utils')
# Add this path to the sys.path list
sys.path.append(f"{parent_directory}")

from config import ModelParameters, TrainingParameters

def make_net(model, n_layers, hidden_units, output_units, net_type='GRU'):
    if net_type=='GRU':
        for i in range(n_layers):
            model.add(GRU(units=hidden_units,
                      return_sequences=True,
                      name=f'GRU_{i + 1}'))
    else:
        for i in range(n_layers):
            model.add(LSTM(units=hidden_units,
                      return_sequences=True,
                      name=f'LSTM_{i + 1}'))

    model.add(Dense(units=output_units,
                    activation='sigmoid',
                    name='OUT'))
    return model


class TimeGAN():

    __MODEL__ = 'TimeGAN'

    def __init__(self, model_parameters: ModelParameters):
        self.model_params = model_parameters
        self.batch_size = model_parameters.batch_size
        self.lr = model_parameters.lr
        self.betas = model_parameters.betas
        self.latent_dim = model_parameters.latent_dim
        self.gamma = model_parameters.gamma
        self.hidden_dim = model_parameters.hidden_dim 
        self.seq_len = None
        self.n_seq = None
        self.num_cols = None

    def fit(self, data, train_arguments: TrainingParameters, num_cols: list[str] | None = None, cat_cols: list[str] | None = None):
        """
        Fits the TimeGAN model.

        Args:
            data: A DataFrame with the data to be synthesized.
            train_arguments: TimeGAN training arguments.
            num_cols: List of numerical columns to be considered.
            cat_cols: List of categorical columns to be considered (optional).
        """
        self.train_params = train_arguments
        self.num_cols = num_cols
        self.seq_len = train_arguments.sequence_length
        self.n_seq = train_arguments.number_sequences

        print(f'num col: {self.num_cols}, sequence length: {self.seq_len} and n sequence: {self.n_seq}')

        # Train the model with the preprocessed data
        self.train(data=data, train_steps=train_arguments.epochs)

        return  


    def sample(self, n_samples: int):
        """
        Samples new data from the TimeGAN.

        Args:
            n_samples: Number of samples to be generated.
        """
        Z_ = next(self.get_batch_noise(size=n_samples))
        records = self.generator(Z_)
        data = []
        for i in range(records.shape[0]):
            data.append(DataFrame(records[i], columns=self.num_cols))
        return data

    def define_gan(self):
        self.generator_aux=Generator(self.hidden_dim).build()
        self.supervisor=Supervisor(self.hidden_dim).build()
        self.discriminator=Discriminator(self.hidden_dim).build()
        self.recovery = Recovery(self.hidden_dim, self.n_seq).build()
        self.embedder = Embedder(self.hidden_dim, self.seq_len).build()

        X = Input(shape=[self.seq_len, self.n_seq], batch_size=self.batch_size, name='RealData')
        Z = Input(shape=[self.seq_len, self.n_seq], batch_size=self.batch_size, name='RandomNoise')

        #--------------------------------
        # Building the AutoEncoder
        #--------------------------------
        H = self.embedder(X)
        X_tilde = self.recovery(H)

        self.autoencoder = Model(inputs=X, outputs=X_tilde)

        #---------------------------------
        # Adversarial Supervise Architecture
        #---------------------------------
        E_Hat = self.generator_aux(Z)
        H_hat = self.supervisor(E_Hat)
        Y_fake = self.discriminator(H_hat)

        self.adversarial_supervised = Model(inputs=Z,
                                       outputs=Y_fake,
                                       name='AdversarialSupervised')

        #---------------------------------
        # Adversarial architecture in latent space
        #---------------------------------
        Y_fake_e = self.discriminator(E_Hat)

        self.adversarial_embedded = Model(inputs=Z,
                                    outputs=Y_fake_e,
                                    name='AdversarialEmbedded')
        # ---------------------------------
        # Synthetic data generation
        # ---------------------------------
        X_hat = self.recovery(H_hat)
        self.generator = Model(inputs=Z,
                            outputs=X_hat,
                            name='FinalGenerator')

        # --------------------------------
        # Final discriminator model
        # --------------------------------
        Y_real = self.discriminator(H)
        self.discriminator_model = Model(inputs=X,
                                         outputs=Y_real,
                                         name="RealDiscriminator")

        # ----------------------------
        # Define the loss functions
        # ----------------------------
        self._mse=MeanSquaredError()
        self._bce=BinaryCrossentropy()


    @function
    def train_autoencoder(self, x, opt):
        with GradientTape() as tape:
            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self._mse(x, x_tilde)
            e_loss_0 = 10 * sqrt(embedding_loss_t0)

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss_0, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return sqrt(embedding_loss_t0)

    @function
    def train_supervisor(self, x, opt):
        with GradientTape() as tape:
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self._mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])

        var_list = self.supervisor.trainable_variables + self.generator.trainable_variables
        gradients = tape.gradient(generator_loss_supervised, var_list)
        apply_grads = [(grad, var) for (grad, var) in zip(gradients, var_list) if grad is not None]
        opt.apply_gradients(apply_grads)
        return generator_loss_supervised

    @function
    def train_embedder(self,x, opt):
        with GradientTape() as tape:
            # Supervised Loss
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self._mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])

            # Reconstruction Loss
            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self._mse(x, x_tilde)
            e_loss = 10 * sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return sqrt(embedding_loss_t0)

    def discriminator_loss(self, x, z):
        # Loss on false negatives
        y_real = self.discriminator_model(x)
        discriminator_loss_real = self._bce(y_true=ones_like(y_real),
                                            y_pred=y_real)

        # Loss on false positives
        y_fake = self.adversarial_supervised(z)
        discriminator_loss_fake = self._bce(y_true=zeros_like(y_fake),
                                            y_pred=y_fake)

        y_fake_e = self.adversarial_embedded(z)
        discriminator_loss_fake_e = self._bce(y_true=zeros_like(y_fake_e),
                                              y_pred=y_fake_e)
        return (discriminator_loss_real +
                discriminator_loss_fake +
                self.gamma * discriminator_loss_fake_e)

    @staticmethod
    def calc_generator_moments_loss(y_true, y_pred):
        y_true_mean, y_true_var = nn.moments(x=y_true, axes=[0])
        y_pred_mean, y_pred_var = nn.moments(x=y_pred, axes=[0])
        g_loss_mean = reduce_mean(abs(y_true_mean - y_pred_mean))
        g_loss_var = reduce_mean(abs(sqrt(y_true_var + 1e-6) - sqrt(y_pred_var + 1e-6)))
        return g_loss_mean + g_loss_var

    @function
    def train_generator(self, x, z, opt):
        with GradientTape() as tape:
            # Compute the generator loss using adversarial supervision
            y_fake = self.adversarial_supervised(z)
            generator_loss_unsupervised = self._bce(y_true=ones_like(y_fake),
                                                    y_pred=y_fake)

            y_fake_e = self.adversarial_embedded(z)
            generator_loss_unsupervised_e = self._bce(y_true=ones_like(y_fake_e),
                                                      y_pred=y_fake_e)
            #  Supervised loss comparing the embedder's output with the supervisor's output
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self._mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])

            # Generate data using the generator
            x_hat = self.generator(z)
            generator_moment_loss = self.calc_generator_moments_loss(x, x_hat)

            # print(f'x and x_hat: {x} and {x_hat}')
            # print(f'print type {type(x)} and {type(x_hat)}')
            # print(f'generator_moment_loss {generator_moment_loss} and its type {type(generator_moment_loss)}')

            # tf.print("generator moment loss", generator_moment_loss)

            # Combine losses
            generator_loss = (generator_loss_unsupervised +
                              generator_loss_unsupervised_e +
                              100 * sqrt(generator_loss_supervised) +
                              100 * generator_moment_loss)
            
            # print(f'generator_loss {generator_moment_loss} and its type {type(generator_moment_loss)}')
            # tf.print("generator loss", generator_loss)

        var_list = self.generator_aux.trainable_variables + self.supervisor.trainable_variables
        gradients = tape.gradient(generator_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss

    @function
    def train_discriminator(self, x, z, opt):
        with GradientTape() as tape:
            discriminator_loss = self.discriminator_loss(x, z)

        var_list = self.discriminator.trainable_variables
        gradients = tape.gradient(discriminator_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return discriminator_loss

    def get_batch_data(self, data, n_windows):
        data = convert_to_tensor(data, dtype=float32)
        return iter(tfdata.Dataset.from_tensor_slices(data)
                                .shuffle(buffer_size=n_windows)
                                .batch(self.batch_size).repeat())

    def _generate_noise(self):
        while True:
            yield np.random.uniform(low=0, high=1, size=(self.seq_len, self.n_seq))

    def get_batch_noise(self, size=None):
        return iter(tfdata.Dataset.from_generator(self._generate_noise, output_types=float32)
                                .batch(self.batch_size if size is None else size)
                                .repeat())

    def train(self, data, train_steps):
        # Assemble the model
        self.define_gan()
        self.g_lr = self.lr
        self.d_lr = self.lr
        self.seq_len = self.seq_len

        ## Embedding network training
        autoencoder_opt = Adam(learning_rate=self.g_lr)
        for _ in tqdm(range(train_steps), desc='Embedding network training'):
            X_ = next(self.get_batch_data(data, n_windows=len(data)))
            step_e_loss_t0 = self.train_autoencoder(X_, autoencoder_opt)
            # tqdm.write(f"Step Loss Embedding: {step_e_loss_t0:.4f}")

        ## Supervised Network training
        supervisor_opt = Adam(learning_rate=self.g_lr)
        for _ in tqdm(range(train_steps), desc='Supervised network training'):
            X_ = next(self.get_batch_data(data, n_windows=len(data)))
            step_g_loss_s = self.train_supervisor(X_, supervisor_opt)
            # tqdm.write(f"Step Loss Supervisor: {step_g_loss_s:.4f}")


        ## Joint training
        generator_opt = Adam(learning_rate=self.g_lr)
        embedder_opt = Adam(learning_rate=self.g_lr)
        discriminator_opt = Adam(learning_rate=self.d_lr)

        step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0
        for _ in tqdm(range(train_steps), desc='Joint networks training'):

            #Train the generator (k times as often as the discriminator)
            # Here k=2
            for _ in range(2):
                X_ = next(self.get_batch_data(data, n_windows=len(data)))
                Z_ = next(self.get_batch_noise())
                # --------------------------
                # Train the generator
                # --------------------------
                step_g_loss_u, step_g_loss_s, step_g_loss_v = self.train_generator(X_, Z_, generator_opt)

                # --------------------------
                # Train the embedder
                # --------------------------
                step_e_loss_t0 = self.train_embedder(X_, embedder_opt)

                # tqdm.write(f"Gen Loss Unsupervised: {step_g_loss_u:.4f}, Supervised: {step_g_loss_s:.4f}, Moment: {step_g_loss_v:.4f}")


            X_ = next(self.get_batch_data(data, n_windows=len(data)))
            Z_ = next(self.get_batch_noise())
            step_d_loss = self.discriminator_loss(X_, Z_)
            # tqdm.write(f"Discriminator Loss: {step_d_loss:.4f}")
            if step_d_loss > 0.15:
                step_d_loss = self.train_discriminator(X_, Z_, discriminator_opt)


class Generator(Model):
    def __init__(self, hidden_dim, net_type='GRU'):
        self.hidden_dim = hidden_dim
        self.net_type = net_type

    def build(self):
        model = Sequential(name='Generator')
        model = make_net(model,
                         n_layers=3,
                         hidden_units=self.hidden_dim,
                         output_units=self.hidden_dim,
                         net_type=self.net_type)
        return model

class Discriminator(Model):
    def __init__(self, hidden_dim, net_type='GRU'):
        self.hidden_dim = hidden_dim
        self.net_type=net_type

    def build(self):
        model = Sequential(name='Discriminator')
        model = make_net(model,
                         n_layers=3,
                         hidden_units=self.hidden_dim,
                         output_units=1,
                         net_type=self.net_type)
        return model

class Recovery(Model):
    def __init__(self, hidden_dim, n_seq):
        self.hidden_dim=hidden_dim
        self.n_seq=n_seq
        return

    def build(self):
        recovery = Sequential(name='Recovery')
        recovery = make_net(recovery,
                            n_layers=3,
                            hidden_units=self.hidden_dim,
                            output_units=self.n_seq)
        return recovery

class Embedder(Model):

    def __init__(self, hidden_dim, seq_len):
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        return

    def build(self):
        embedder = Sequential(name='Embedder')
        embedder.add(Input(shape=(self.seq_len, self.hidden_dim))) 
        embedder = make_net(embedder,
                            n_layers=3,
                            hidden_units=self.hidden_dim,
                            output_units=self.hidden_dim)
        return embedder

class Supervisor(Model):
    def __init__(self, hidden_dim):
        self.hidden_dim=hidden_dim

    def build(self):
        model = Sequential(name='Supervisor')
        model = make_net(model,
                         n_layers=2,
                         hidden_units=self.hidden_dim,
                         output_units=self.hidden_dim)
        return model