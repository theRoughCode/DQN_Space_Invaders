from gym import wrappers
from dqn import Agent
from utils import preprocess, plotLearning
import gym

LOAD_MODEL = True
SAVE_MODEL = False

scores = []
epsHistory = []
numGames = 1
batch_size = 32
n_actions = 6
input_dims = (185, 95)
crop_start = (15, 30)
crop_end = (200, 125)
starting_epsilon = 0.05 if LOAD_MODEL else 1.0

env = gym.make('SpaceInvaders-v0')
brain = Agent(gamma=0.95, epsilon=0.05, lr=0.003, input_dims=input_dims,
              batch_size=batch_size, n_actions=n_actions, max_mem_size=5000, save_path='models/')

if LOAD_MODEL:
    brain.load()
else:
    # load memory with random games
    while brain.mem_cntr < brain.mem_size:
      observation = env.reset()
      observation = preprocess(observation, crop_start, crop_end)
      done = False
      while not done:
        # 0 no action, 1 fire, 2 move right, 3 move left, 4 move right fire, 5 move left fire
        action = env.action_space.sample()
        observation_, reward, done, info = env.step(action)
        observation_ = preprocess(observation_, crop_start, crop_end)
        if done and info['ale.lives'] == 0:
          reward = -100
        brain.store_transition(observation, action, reward, observation_, int(done))
        observation = observation_
    print('done initializing memory')

# uncomment the line below to record every episode.
# env = wrappers.Monitor(env, "tmp/space-invaders-1", video_callable=lambda episode_id: True, force=True)
for i in range(numGames):
  print('starting game ', i+1, 'epsilon: %.4f' % brain.epsilon)
  epsHistory.append(brain.epsilon)
  done = False
  observation = env.reset()
  observation = preprocess(observation, crop_start, crop_end)
  score = 0
  while not done:
    action = brain.choose_action(observation)
    observation_, reward, done, info = env.step(action)
    score += reward
    observation_ = preprocess(observation_, crop_start, crop_end)
    if done and info['ale.lives'] == 0:
      reward = -100
    brain.store_transition(observation, action, reward, observation_, int(done))
    observation = observation_
    brain.learn()
    env.render()
  scores.append(score)
  print('score:',score)
x = [i+1 for i in range(numGames)]
fileName = str(numGames) + 'Games' + 'Gamma' + str(brain.gamma) + \
            'Alpha' + str(brain.lr) + 'Memory' + str(brain.mem_size)+ '.png'
plotLearning(x, scores, epsHistory, 'plots/' + fileName)

if SAVE_MODEL:
    brain.save()