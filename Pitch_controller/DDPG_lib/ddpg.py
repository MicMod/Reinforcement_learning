import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from DDPG_lib.buffer import ReplayBuffer
from DDPG_lib.networks import ActorNetwork, CriticNetwork, get_actor, get_critic

from keras.models import load_model

class Agent:
    def __init__(self, input_dims, lr_a=0.001, lr_c=0.002, env=None,
            gamma=0.99, n_actions=1, max_size=1000000, tau=0.005, 
            fc1=400, fc2=300, batch_size=64, std_noise=0.2, action_bound_range=np.deg2rad(25), chkpt_dir='DDPG_checkpoints', model_dir='DDPG_models'):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.state_dim = input_dims
        
        self.action_bound_range = action_bound_range
        self.max_action = action_bound_range
        self.min_action = -action_bound_range
        self.std_noise = std_noise
        
        self.chkpt_dir=chkpt_dir
        self.model_dir = model_dir

        self.actor = get_actor()
        self.critic = get_critic()
        self.target_actor = get_actor()
        self.target_critic = get_critic()
        
        self.actor.compile(optimizer=Adam(learning_rate=lr_a))
        self.critic.compile(optimizer=Adam(learning_rate=lr_c))
        self.target_actor.compile(optimizer=Adam(learning_rate=lr_a))
        self.target_critic.compile(optimizer=Adam(learning_rate=lr_c))

        self.update_network_parameters(tau=self.tau)
        self.ensure_dir(dir=self.chkpt_dir)
        self.ensure_dir(dir=self.model_dir)
        
    def ensure_dir(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir) 

    def update_network_parameters(self, tau=None):
        tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_weights(self, episode):
        print('... saving weights ...')
        self.actor.save_weights(os.path.join(self.chkpt_dir, "actor_weights_{}.h5".format(episode)))
        self.target_actor.save_weights(os.path.join(self.chkpt_dir, "target_actor_weights_{}.h5".format(episode)))
        self.critic.save_weights(os.path.join(self.chkpt_dir, "critic_weights_{}.h5".format(episode)))
        self.target_critic.save_weights(os.path.join(self.chkpt_dir, "target_critic_weights_{}.h5".format(episode)))

    def load_weights(self, episode):
        print('... loading weights ...')
        self.actor.load_weights(os.path.join(self.chkpt_dir, "actor_weights_{}.h5".format(episode)))
        self.target_actor.load_weights(os.path.join(self.chkpt_dir, "target_actor_weights_{}.h5".format(episode)))
        self.critic.load_weights(os.path.join(self.chkpt_dir, "critic_weights_{}.h5".format(episode)))
        self.target_critic.load_weights(os.path.join(self.chkpt_dir, "target_critic_weights_{}.h5".format(episode)))

    def save_models(self):
        print('... saving models ...')
        self.actor.save(os.path.join(self.model_dir, 'actor.h5'))
        self.target_actor.save(os.path.join(self.model_dir, 'target_actor.h5'))
        self.critic.save(os.path.join(self.model_dir, 'critic.h5'))
        self.target_critic.save(os.path.join(self.model_dir, 'target_critic.h5'))

    def load_models(self):
        print('... loading models ...')
        self.actor.load_model(os.path.join(self.model_dir, 'actor.h5'))
        self.target_actor.load_model(os.path.join(self.model_dir, 'target_actor.h5'))
        self.critic.load_model(os.path.join(self.model_dir, 'critic.h5'))
        self.target_critic.load_model(os.path.join(self.model_dir, 'target_critic.h5'))
        
    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.action_bound_range*self.actor(state)
        
        if not evaluate:
            actions +=  tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.std_noise)
            
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        new_sates = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_sates)
            critic_value_ = self.target_critic([
                                new_sates, target_actions], 1)
            critic_value = self.critic([states, actions], 1)
            target = reward + self.gamma*critic_value_*(1-done)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                                            self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
                                            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic([states, new_policy_actions])
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, 
                                    self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()
