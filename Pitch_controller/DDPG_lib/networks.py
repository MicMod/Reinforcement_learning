import os
import numpy as np 
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.initializers import RandomUniform as RU

def get_actor(fc1_dims=400, fc2_dims=300, state_dim=4,  n_actions=1):
    
    fc1 = Dense(fc1_dims, activation='relu', kernel_initializer=RU(-1/np.sqrt(state_dim),1/np.sqrt(state_dim)))
    fc2 = Dense(fc2_dims, activation='relu', kernel_initializer=RU(-1/np.sqrt(fc1_dims),1/np.sqrt(fc1_dims)))
    mu = Dense(n_actions, activation='tanh', kernel_initializer=RU(-0.003,0.003))
        
    inputs = Input(shape=(state_dim))
    out = fc1(inputs)
    out = fc2(out)
    outputs = mu(out)        

    model = tf.keras.Model(inputs, outputs)
    return model

def get_critic(fc1_dims=400, fc2_dims=300, state_dim=4,  n_actions=1):
    
    fc1 = Dense(fc1_dims, activation='relu', kernel_initializer=RU(-1/np.sqrt(state_dim),1/np.sqrt(state_dim)))
    fc2 = Dense(fc2_dims, activation='relu', kernel_initializer=RU(-1/np.sqrt(fc1_dims),1/np.sqrt(fc1_dims)))
    q = Dense(1, activation=None, kernel_initializer=RU(-0.003,0.003))
    state_input = Input(shape=(state_dim))
    action_input = Input(shape=(n_actions))
    
    out = fc1(state_input)
    out = concatenate([out, action_input])
    out = fc2(out)
    output = q(out)  

    model = tf.keras.Model([state_input, action_input], output)
    
    return model
    