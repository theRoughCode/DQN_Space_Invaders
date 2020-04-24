import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
      super(DeepQNetwork, self).__init__()
      self.input_dims = input_dims[0] * input_dims[1]
      self.fc1_dims = fc1_dims
      self.fc2_dims = fc2_dims
      self.n_actions = n_actions
      self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
      self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
      self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

      self.optimizer = optim.Adam(self.parameters(), lr=lr)
      self.loss = nn.MSELoss()
      self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
      self.to(self.device)

    def forward(self, state):
      # state: (B, H, W)
      state = state.view(-1, self.input_dims) #(B, H*W)
      # layer 1
      x = self.fc1(state)                     # (B, fc1_dims)
      x = F.relu(x)
      # layer 2
      x = self.fc2(x)                         # (B, fc2_dims)
      x = F.relu(x)
      # output layer
      actions = self.fc3(x)                   # (B, n_actions)

      return actions

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
            max_mem_size=100000, eps_end=0.05, eps_dec=5e-4, save_path='models/'):
      self.gamma = gamma
      self.epsilon = epsilon
      self.eps_min = eps_end
      self.eps_dec = eps_dec
      self.lr = lr
      self.action_space = [i for i in range(n_actions)]
      self.mem_size = max_mem_size
      self.batch_size = batch_size
      self.mem_cntr = 0
      self.iter_cntr = 0
      self.replace_target = 100
      self.save_path = save_path

      self.Q_eval = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                  fc1_dims=256, fc2_dims=256)
      self.Q_next = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                  fc1_dims=256, fc2_dims=256)

      self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
      self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
      self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
      self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
      self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def save(self):
      T.save(self.Q_eval.state_dict(), self.save_path + 'q_eval.pt')
      T.save(self.Q_next.state_dict(), self.save_path + 'q_next.pt')

    def load(self):
      self.Q_eval.load_state_dict(T.load(self.save_path + 'q_eval.pt'))
      self.Q_next.load_state_dict(T.load(self.save_path + 'q_next.pt'))

    def store_transition(self, state, action, reward, state_, terminal):
      # store transition in memory queue
      index = self.mem_cntr % self.mem_size
      self.state_memory[index] = state
      self.new_state_memory[index] = state_
      self.reward_memory[index] = reward
      self.action_memory[index] = action
      self.terminal_memory[index] = terminal

      self.mem_cntr += 1

    def choose_action(self, observation):
      # e-greedy strategy
      if np.random.random() > self.epsilon:
        state = T.tensor(observation).float().to(self.Q_eval.device)
        actions = self.Q_eval.forward(state)
        action = T.argmax(actions[0]).item()
      else:
        action = np.random.choice(self.action_space)
      return action

    def learn(self):
      if self.mem_cntr < self.batch_size:
        return

      self.Q_eval.optimizer.zero_grad()
      
      # Experience learning: sample batches from memory
      max_mem = min(self.mem_cntr, self.mem_size)
      batch = np.random.choice(max_mem, self.batch_size, replace=False)
      batch_index = np.arange(self.batch_size, dtype=np.int32)
      # Retrieve relevant batched inputs
      state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
      new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
      action_batch = self.action_memory[batch]
      reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
      terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

      # Feed into networks
      q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
      # TODO: Check if this should be Q_next
      q_next = self.Q_next.forward(new_state_batch)
      q_next[terminal_batch] = 0.0

      # Get target values
      q_target = reward_batch + self.gamma*T.max(q_next,dim=1)[0]

      # Compute loss and backpropagate
      loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
      loss.backward()
      self.Q_eval.optimizer.step()

      self.iter_cntr += 1
      self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                      else self.eps_min

      if self.iter_cntr % self.replace_target == 0:
        self.Q_next.load_state_dict(self.Q_eval.state_dict())