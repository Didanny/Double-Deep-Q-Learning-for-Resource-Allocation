from __future__ import print_function, division
import os
import time
import torch
import random
import copy
import numpy as np
from Environment import *
from base import BaseModel
from replay_memory import ReplayMemory
from utils import save_pkl, load_pkl
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

class QHD_Model(object):
    def __init__(self,
                dimension=10000,
                n_actions=2,
                n_obs=4,
                epsilon=1.0,
                epsilon_decay=0.995,
                minimum_epsilon=0.01,
                reward_decay=0.9,
                mem=70, #70#50#200
                batch=20, #20#10#50
                lr = 0.05#0.05 #0.035
                ):
        self.D = dimension
        self.n_actions = n_actions
        self.n_obs = n_obs
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.minimum_epsilon = minimum_epsilon
        self.reward_decay = reward_decay
        self.mem = mem
        self.batch = batch
        self.lr = lr
        
        self.logs = [] # temp log for current episode
        self.episode_logs = [] # logs for past episodes
        self.model = []

        for a in range(n_actions):
            self.model.append(np.zeros(dimension, dtype=complex))
        self.s_hdvec = []
        for n in range(self.n_obs):
            self.s_hdvec.append(np.random.normal(0, 1, dimension))
        self.bias = np.random.uniform(0,2*np.pi, size=dimension)

        self.s_hdvec = np.array(self.s_hdvec)
        self.bias = np.array(self.bias)
        self.model = np.array(self.model)
        self.delay_model = copy.deepcopy(self.model)

        self.s_hdvec = torch.from_numpy(self.s_hdvec)
        self.bias = torch.from_numpy(self.bias)
        self.model = torch.from_numpy(self.model)
        self.delay_model = torch.from_numpy(self.delay_model)
        
        self.model_update_counter = 0
        self.tau = 1 # for cartpole 
    
    def store_transition(self, s, a, r, n_s):
        self.logs.append((s,a,r,n_s))
        if len(self.logs) > 100000: # if the mem is full, POP
            self.logs.pop(0)

    def update_delay_model(self):
        self.delay_model = self.model.detach().clone()

    def save_model(self):
        torch.save(self.model, 'model.pt')
        torch.save(self.delay_model, 'delay_model.pt')
        torch.save(self.bias, 'bias.pt')
        torch.save(self.s_hdvec, 's_hdvec.pt')
    
    def choose_action(self, observation): # observation should be numpy ndarray
        if (random.random() <= self.epsilon):
            self.action = random.randint(0, self.n_actions-1)
        else:
            q_values = list()
            for action in range(self.n_actions):
                q_values.append(self.value(action, observation))
            self.action = np.argmax(q_values)
        return self.action

    def q_values(self, observation, delay=False):
        q_values = list()
        for action in range(self.n_actions):
            q_values.append(self.value(action, observation, delay))
        return q_values
    
    def value(self, action, observation, delay=False):
        ## Encoding
        encoded = torch.exp(1j* ((observation @ self.s_hdvec)+self.bias))
        if delay == True:
            q_value = torch.real((torch.conj(encoded) @ self.delay_model[action])/self.D)
        else:
            q_value = torch.real((torch.conj(encoded) @ self.model[action])/self.D)
        return q_value
    
    def feedback(self):
        ## Update the delayed model
        self.model_update_counter += 1
        if self.model_update_counter > self.tau:
            self.delay_model = copy.deepcopy(self.model)
            self.model_update_counter = 0

        self.episode_logs.append(self.logs)
        if len(self.episode_logs) > self.mem: # if the mem is full, POP
            self.episode_logs.pop(0)

        for iter in range(15): #15
            if len(self.episode_logs) < self.batch:
                indexs = list(range(len(self.episode_logs)))
            else:
                indexs = random.sample(list(range(len(self.episode_logs))), self.batch)
            for i in indexs:
                episode_logs = self.episode_logs[i]
                if len(episode_logs) < 1:
                    idx = list(range(len(episode_logs)))
                else:
                    idx = random.sample(list(range(len(episode_logs))), 1) + [len(episode_logs)-1]
                for j in idx:
                    log = episode_logs[j]
                    (obs, action, reward, next_obs) = log
                    y_pred = self.value(action, obs)
                    q_list = []
                    for a_ in range(self.n_actions):
                        q_list.append(self.value(a_, next_obs, True))
                    y_true = reward + self.reward_decay*max(q_list)
                    encoded = np.exp(1j* (np.matmul(obs, self.s_hdvec[action])+self.bias[action]))
                    # model_size = np.linalg.norm(self.model[action])/self.D
                    # if model_size != 0:
                    #     print(action, model_size)
                    #    self.model[action] += self.lr * model_size * (y_true-y_pred) * encoded
                    #else:
                    self.model[action] += self.lr * (y_true-y_pred) * encoded
                    #print(y_true-y_pred)

    def new_feedback(self): # for stepwise model update
        for iter in range(1):
            if len(self.logs) < 15: #5 for acrobot
                logs = self.logs
            else:
                logs = random.sample(self.logs, 15)
            for k in range(1):
                for log in logs:
                    (obs, action, reward, next_obs) = log
                    y_pred = self.value(action, obs)
                    # q_list = []
                    # for a_ in range(self.n_actions):
                    #     q_list.append(self.value(a_, next_obs, True))
                    q_list = self.q_values(next_obs, True)
                    y_true = reward + self.reward_decay*max(q_list)
                    encoded = torch.exp(1j* ((obs @ self.s_hdvec[action])+self.bias[action]))
                    # model_size = np.linalg.norm(self.model[action])/self.D
                    # if model_size != 0:
                    #     print(action, model_size)
                    #    self.model[action] += self.lr * model_size * (y_true-y_pred) * encoded
                    #else:
                    self.model[action] += self.lr * (y_true-y_pred) * encoded
                    #print(y_true-y_pred)
                    return 

    def train_on_sample(self, obs, action, reward, next_obs):
        y_pred = self.value(action, obs)
        q_list = []
        encoded = torch.exp(1j* ((obs @ self.s_hdvec)+self.bias))
        encoded_ = torch.exp(1j* ((next_obs @ self.s_hdvec)+self.bias))
        for a_ in range(self.n_actions):
            q_list.append(torch.real((torch.conj(encoded_) @ self.delay_model[a_])/self.D))
        y_true = reward + self.reward_decay*max(q_list)
        self.model[action] += self.lr * (y_true-y_pred) * encoded
        #print(y_true-y_pred)
        return y_true-y_pred, y_true

class Agent(BaseModel):
    def __init__(self, config, environment, sess):
        self.sess = sess
        self.weight_dir = 'weight'        
        self.env = environment
        #self.history = History(self.config)
        model_dir = './Model/a.model'
        self.memory = ReplayMemory(model_dir) 
        self.max_step = 100000
        self.train_steps = 10_000
        self.batch_size = 2000
        self.RB_number = 20
        self.num_vehicle = 20
        self.num_vehicle_train = 20
        print('-------------------------------------------')
        print(self.num_vehicle)
        print('-------------------------------------------')
        self.action_all_with_power = np.zeros([self.num_vehicle, 3, 2],dtype = 'int32')   # this is actions that taken by V2V links with power
        self.action_all_with_power_training = np.zeros([self.num_vehicle_train, 3, 2],dtype = 'int32')   # this is actions that taken by V2V links with power
        self.reward = []
        self.learning_rate = 0.01
        self.learning_rate_minimum = 0.0001
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 500000
        self.target_q_update_step = 100
        self.discount = 0.5
        self.double_q = True
        self.qhd_model = QHD_Model(dimension=10_000, n_actions=60, n_obs=82, epsilon=1, epsilon_decay=0.995, minimum_epsilon=0.010,
                                   reward_decay=0.9, mem=70, batch=20, lr=0.035)
        print("------------")
        print(self.double_q)
        print("------------")
        # self.build_dqn()          
        self.V2V_number = 3 * len(self.env.vehicles)    # every vehicle need to communicate with 3 neighbors  
        self.training = True
        #self.actions_all = np.zeros([len(self.env.vehicles),3], dtype = 'int32')
    def merge_action(self, idx, action):
        self.action_all_with_power[idx[0], idx[1], 0] = action % self.RB_number
        self.action_all_with_power[idx[0], idx[1], 1] = int(np.floor(action/self.RB_number))
    def get_state(self, idx):
    # ===============
    #  Get State from the environment
    # =============
        vehicle_number = len(self.env.vehicles)
        V2V_channel = (self.env.V2V_channels_with_fastfading[idx[0],self.env.vehicles[idx[0]].destinations[idx[1]],:] - 80)/60
        V2I_channel = (self.env.V2I_channels_with_fastfading[idx[0], :] - 80)/60
        V2V_interference = (-self.env.V2V_Interference_all[idx[0],idx[1],:] - 60)/60
        NeiSelection = np.zeros(self.RB_number)
        for i in range(3):
            for j in range(3):
                if self.training:
                    NeiSelection[self.action_all_with_power_training[self.env.vehicles[idx[0]].neighbors[i], j, 0 ]] = 1
                else:
                    NeiSelection[self.action_all_with_power[self.env.vehicles[idx[0]].neighbors[i], j, 0 ]] = 1
                   
        for i in range(3):
            if i == idx[1]:
                continue
            if self.training:
                if self.action_all_with_power_training[idx[0],i,0] >= 0:
                    NeiSelection[self.action_all_with_power_training[idx[0],i,0]] = 1
            else:
                if self.action_all_with_power[idx[0],i,0] >= 0:
                    NeiSelection[self.action_all_with_power[idx[0],i,0]] = 1
        time_remaining = np.asarray([self.env.demand[idx[0],idx[1]] / self.env.demand_amount])
        load_remaining = np.asarray([self.env.individual_time_limit[idx[0],idx[1]] / self.env.V2V_limit])
        #print('shapes', time_remaining.shape,load_remaining.shape)
        return np.concatenate((V2I_channel, V2V_interference, V2V_channel, NeiSelection, time_remaining, load_remaining))#,time_remaining))
        #return np.concatenate((V2I_channel, V2V_interference, V2V_channel, time_remaining, load_remaining))#,time_remaining))
    def predict(self, s_t,  step, test_ep = False):
        # ==========================
        #  Select actions
        # ======================
        # print("===================================")
        # print(s_t.shape)
        # print("===================================")
        ep = 1/(step/1000000 + 1)
        # if (step % 50 != 0):
        #     print("epsilon = {0}".format(ep))
        if random.random() < ep and test_ep == False:   # epsilon to balance the exporation and exploition
            action = np.random.randint(60)
        else:    
            # print('s_t Type:')
            # print(torch.from_numpy(s_t).type())
            action = np.argmax(self.qhd_model.q_values(torch.from_numpy(s_t)))       
            # action =  self.q_action.eval({self.s_t:[s_t]})[0] 
        return action
    def observe(self, prestate, state, reward, action):
        # -----------
        # Collect Data for Training 
        # ---------
        self.memory.add(prestate, state, reward, action) # add the state and the action and the reward to the memory
        #print(self.step)
        l = "no loss"
        if self.step > 0:
            if self.step % 50 == 0:
                #print('Training')
                l = self.q_learning_mini_batch_qhd()            # training a mini batch
                self.qhd_model.save_model()
                # self.save_weight_to_pkl()
            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                print("Update Target Q network:")
                self.update_target_qhd_model()           # ?? what is the meaning ??
            return l
    def train(self):        
        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0.,0.,0.
        max_avg_ep_reward = 0
        ep_reward, actions = [], []        
        mean_big = 0
        number_big = 0
        mean_not_big = 0
        number_not_big = 0
        self.env.new_random_game(self.num_vehicle_train)
        for self.step in (range(0, self.train_steps)): # need more configuration
            if self.step == 0:                   # initialize set some varibles
                num_game, self.update_count,ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_reward, actions = [], []               
                
            # prediction
            # action = self.predict(self.history.get())
            if (self.step % 2000 == 1):
                self.env.new_random_game(self.num_vehicle_train)
            print(self.step)
            state_old = self.get_state([0,0])
            #print("state", state_old)
            self.training = True
            for k in range(1):
                if (self.step % 50 == 0 and self.step > 0):
                    pbar = tqdm(total=self.num_vehicle_train)
                for i in range(len(self.env.vehicles)):              
                    for j in range(3): 
                        state_old = self.get_state([i,j]) 
                        action = self.predict(state_old, self.step)                    
                        #self.merge_action([i,j], action)   
                        self.action_all_with_power_training[i, j, 0] = action % self.RB_number
                        self.action_all_with_power_training[i, j, 1] = int(np.floor(action/self.RB_number))                                                    
                        reward_train = self.env.act_for_training(self.action_all_with_power_training, [i,j]) 
                        state_new = self.get_state([i,j]) 
                        l = self.observe(state_old, state_new, reward_train, action)
                    if (self.step % 50 == 0 and self.step > 0):
                        pbar.set_description("Training Progress:")
                        # pbar.write("loss is {0}".format(l))
                        pbar.update()
            if (self.step % 2000 == 0) and (self.step > 0):
                # testing 
                self.training = False
                number_of_game = 5
                # if (self.step % self.train_steps == 0) and (self.step > 0):
                #     number_of_game = 50 
                # if (self.step == 38000):
                #     number_of_game = 100               
                V2I_Rate_list = np.zeros(number_of_game)
                Fail_percent_list = np.zeros(number_of_game)
                for game_idx in range(number_of_game):
                    self.env.new_random_game(self.num_vehicle)
                    test_sample = 200
                    Rate_list = []
                    print('test game idx:', game_idx)
                    # pbar = tqdm(total = test_sample)
                    for k in range(test_sample):
                        action_temp = self.action_all_with_power.copy()
                        for i in range(len(self.env.vehicles)):
                            self.action_all_with_power[i,:,0] = -1
                            sorted_idx = np.argsort(self.env.individual_time_limit[i,:])          
                            for j in sorted_idx:                   
                                state_old = self.get_state([i,j])
                                action = self.predict(state_old, self.step, True)
                                self.merge_action([i,j], action)
                            if i % (len(self.env.vehicles)/10) == 1:
                                action_temp = self.action_all_with_power.copy()
                                reward, percent = self.env.act_asyn(action_temp) #self.action_all)            
                                Rate_list.append(np.sum(reward))
                        #print("actions", self.action_all_with_power)
                        # pbar.set_description("Replay Progress:")
                        # pbar.update()
                    V2I_Rate_list[game_idx] = np.mean(np.asarray(Rate_list))
                    Fail_percent_list[game_idx] = percent
                    #print("action is", self.action_all_with_power)
                    print('failure probability is, ', percent)
                    #print('action is that', action_temp[0,:])
            #print("OUT")
                # self.save_weight_to_pkl()
                print ('The number of vehicle is ', len(self.env.vehicles))
                print ('Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
                print('Mean of Fail percent is that ', np.mean(Fail_percent_list))                   
                #print('Test Reward is ', np.mean(test_result))
             
                  
    def q_learning_mini_batch_qhd(self):

        s_t, s_t_plus_1, action, reward = self.memory.sample()
        # print('s_t shape:')
        # print(s_t.shape)
        # print('s_t_plus_1 shape:')
        # print(s_t_plus_1.shape)
        # print('action shape:')
        # print(action.shape)
        # print('reward shape:')
        # print(reward.shape)

        t = time.time()
        losses = torch.zeros(self.batch_size)
        qs = torch.zeros(self.batch_size)
        if self.double_q:
            # print('s_t_plus_1 Type:')
            # print(torch.from_numpy(s_t_plus_1[0]).shape)
            # pbar = tqdm(total=self.batch_size)
            for i in range(self.batch_size):

                losses[i], qs[i] = self.qhd_model.train_on_sample(torch.from_numpy(s_t[i]).to(torch.float64), action[i], reward[i], torch.from_numpy(s_t_plus_1[i]).to(torch.float64))
            
                # pbar.set_description("Training Progress:")
                # pbar.update()

            # print('loss is ', torch.mean(losses))
            self.total_loss += torch.mean(losses)
            self.total_q += torch.mean(qs)
            self.update_count += 1
            return torch.mean(losses)
        else:
            pass

            
    def q_learning_mini_batch(self):

        # Training the DQN model
        # ------ 
        #s_t, action,reward, s_t_plus_1, terminal = self.memory.sample() 
        s_t, s_t_plus_1, action, reward = self.memory.sample()  
        #print() 
        #print('samples:', s_t[0:10], s_t_plus_1[0:10], action[0:10], reward[0:10])        
        t = time.time()        
        if self.double_q:       #double Q learning   
            pred_action = self.q_action.eval({self.s_t: s_t_plus_1})       
            q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({self.target_s_t: s_t_plus_1, self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]})            
            target_q_t =  self.discount * q_t_plus_1_with_pred_action + reward
        else:
            q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})         
            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
            target_q_t = self.discount * max_q_t_plus_1 +reward
        _, q_t, loss,w = self.sess.run([self.optim, self.q, self.loss, self.w], {self.target_q_t: target_q_t, self.action:action, self.s_t:s_t, self.learning_rate_step: self.step}) # training the network
        
        print('loss is ', loss)
        self.total_loss += loss
        self.total_q += q_t.mean()
        self.update_count += 1
            

    def build_dqn(self): 
    # --- Building the DQN -------
        self.w = {}
        self.t_w = {}        
        
        initializer = tf. truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu
        n_hidden_1 = 500
        n_hidden_2 = 250
        n_hidden_3 = 120
        n_input = 82
        n_output = 60
        def encoder(x):
            weights = {                    
                'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],stddev=0.1)),
                'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],stddev=0.1)),
                'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],stddev=0.1)),
                'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_output],stddev=0.1)),
                'encoder_b1': tf.Variable(tf.truncated_normal([n_hidden_1],stddev=0.1)),
                'encoder_b2': tf.Variable(tf.truncated_normal([n_hidden_2],stddev=0.1)),
                'encoder_b3': tf.Variable(tf.truncated_normal([n_hidden_3],stddev=0.1)),
                'encoder_b4': tf.Variable(tf.truncated_normal([n_output],stddev=0.1)),         
            
            }
            layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), weights['encoder_b1']))
            layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']), weights['encoder_b2']))
            layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_h3']), weights['encoder_b3']))
            layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['encoder_h4']), weights['encoder_b4']))
            return layer_4, weights
        with tf.variable_scope('prediction'):
            self.s_t = tf.placeholder('float32',[None, n_input])            
            self.q, self.w = encoder(self.s_t)
            self.q_action = tf.argmax(self.q, dimension = 1)
        with tf.variable_scope('target'):
            self.target_s_t = tf.placeholder('float32', [None, n_input])
            self.target_q, self.target_w = encoder(self.target_s_t)
            self.target_q_idx = tf.placeholder('int32', [None,None], 'output_idx')
            self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)
        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}
            for name in self.w.keys():
                print('name in self w keys', name)
                self.t_w_input[name] = tf.placeholder('float32', self.target_w[name].get_shape().as_list(),name = name)
                self.t_w_assign_op[name] = self.target_w[name].assign(self.t_w_input[name])       
        
        def clipped_error(x):
            try:
                return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
            except:
                return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', None, name='target_q_t')
            self.action = tf.placeholder('int32',None, name = 'action')
            action_one_hot = tf.one_hot(self.action, n_output, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices = 1, name='q_acted')
            self.delta = self.target_q_t - q_acted
            self.global_step = tf.Variable(0, trainable=False)
            self.loss = tf.reduce_mean(tf.square(self.delta), name = 'loss')
            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum, tf.train.exponential_decay(self.learning_rate, self.learning_rate_step, self.learning_rate_decay_step, self.learning_rate_decay, staircase=True))
            self.optim = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss) 
        
        tf.initialize_all_variables().run()
        self.update_target_q_network()

    def update_target_qhd_model(self):
        self.qhd_model.update_delay_model()

    def update_target_q_network(self):    
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})       
        
    def save_weight_to_pkl(self): 
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)
        for name in self.w.keys():
            save_pkl(self.w[name].eval(), os.path.join(self.weight_dir,"%s.pkl" % name))       
    def load_weight_from_pkl(self):
        with tf.variable_scope('load_pred_from_pkl'):
            self.w_input = {}
            self.w_assign_op = {}
            for name in self.w.keys():
                self.w_input[name] = tf.placeholder('float32')
                self.w_assign_op[name] = self.w[name].assign(self.w_input[name])
        for name in self.w.keys():
            self.w_assign_op[name].eval({self.w_input[name]:load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})
        self.update_target_q_network()   
      
    def play(self, n_step = 100, n_episode = 100, test_ep = None, render = False):
        number_of_game = 100
        V2I_Rate_list = np.zeros(number_of_game)
        Fail_percent_list = np.zeros(number_of_game)
        self.load_weight_from_pkl()
        self.training = False


        for game_idx in range(number_of_game):
            self.env.new_random_game(self.num_vehicle)
            test_sample = 200
            Rate_list = []
            print('test game idx:', game_idx)
            print('The number of vehicle is ', len(self.env.vehicles))
            time_left_list = []
            power_select_list_0 = []
            power_select_list_1 = []
            power_select_list_2 = []

            for k in range(test_sample):
                #print(k)
                action_temp = self.action_all_with_power.copy()
                for i in range(len(self.env.vehicles)):
                    self.action_all_with_power[i, :, 0] = -1
                    sorted_idx = np.argsort(self.env.individual_time_limit[i, :])
                    for j in sorted_idx:
                        state_old = self.get_state([i, j])
                        time_left_list.append(state_old[-1])
                        action = self.predict(state_old, 0, True)
                        
                        if state_old[-1] <=0:
                            continue
                        power_selection = int(np.floor(action/self.RB_number))
                        if power_selection == 0:
                            power_select_list_0.append(state_old[-1])

                        if power_selection == 1:
                            power_select_list_1.append(state_old[-1])
                        if power_selection == 2:
                            power_select_list_2.append(state_old[-1])
                        
                        self.merge_action([i, j], action)
                    if i % (len(self.env.vehicles) / 10) == 1:
                        action_temp = self.action_all_with_power.copy()
                        reward, percent = self.env.act_asyn(action_temp)  # self.action_all)
                        Rate_list.append(np.sum(reward))
                # print("actions", self.action_all_with_power)
            
            number_0, bin_edges = np.histogram(power_select_list_0, bins = 10)

            number_1, bin_edges = np.histogram(power_select_list_1, bins = 10)

            number_2, bin_edges = np.histogram(power_select_list_2, bins = 10)


            p_0 = number_0 / (number_0 + number_1 + number_2)
            p_1 = number_1 / (number_0 + number_1 + number_2)
            p_2 = number_2 / (number_0 + number_1 + number_2)
            plt.figure()
            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_0, 'b*-', label='Power Level 23 dB')
            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_1, 'rs-', label='Power Level 10 dB')
            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_2, 'go-', label='Power Level 5 dB')
            plt.xlim([0,0.12])
            plt.xlabel("Time left for V2V transmission (s)")
            plt.ylabel("Probability of power selection")
            plt.legend()
            plt.grid()
            plt.savefig()
            #plt.show()
            
            V2I_Rate_list[game_idx] = np.mean(np.asarray(Rate_list))
            Fail_percent_list[game_idx] = percent

            print('Mean of the V2I rate is that ', np.mean(V2I_Rate_list[0:game_idx] ))
            print('Mean of Fail percent is that ',percent, np.mean(Fail_percent_list[0:game_idx]))
            # print('action is that', action_temp[0,:])

        print('The number of vehicle is ', len(self.env.vehicles))
        print('Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
        print('Mean of Fail percent is that ', np.mean(Fail_percent_list))
        # print('Test Reward is ', np.mean(test_result))
	
	
def main(_):

  up_lanes = [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
  down_lanes = [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]
  left_lanes = [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
  right_lanes = [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]
  width = 750
  height = 1299
  Env = Environ(down_lanes,up_lanes,left_lanes,right_lanes, width, height)
  Env.new_random_game()
  '''
  gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
'''
  with tf.Session(config=tf.ConfigProto()) as sess:
    config = []
    agent = Agent(config, Env, sess)
    #agent.play()
    agent.train()
    # agent.play()

if __name__ == '__main__':
    tf.app.run()
        




