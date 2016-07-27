import gym
import tensorflow as tf
import config
from agent import AgentAsyncDQN
import threading
from network import AsyncDQNetwork
T = 0

def main():

  # Set up game environments (one per thread)
  envs = [gym.make(config.EXPERIMENT) for i in range(config.NUM_THREADS)]
  action_size = envs[0].action_space.n

  network = AsyncDQNetwork(action_size)

  actor_learner_threads = [AgentAsyncDQN(args=(thread_id, envs[thread_id], network)) for thread_id in
                           range(config.NUM_THREADS)]
  for t in actor_learner_threads:
    t.start()

  if config.SHOW_TRAINING:
    for env in envs:
      env.render()

  for t in actor_learner_threads:
    t.join()

  # agent = AgentAsyncDQN(env)


  # experiment = 'Pong-v0'
  # env = gym.make(experiment)
  # print(env.action_space)
  # action_size = env.action_space.n
  # print(env.observation_space)
  #
  # agent = AgentDQN(env, action_size)
  #
  # for i in xrange(config.EPISODES):
  #   obs = env.reset()
  #   agent.set_initial_state(obs)
  #   score = 0
  #   print "Episode ", i
  #   for t in xrange(config.STEPS):
  #     if config.SHOW_TRAINING:
  #       env.render()
  #     action = agent.get_action()
  #     # Execute action a_t and observe reward r_t and observe new observation s_{t+1}
  #     obs, reward, done, _ = env.step(action)
  #     score += reward
  #
  #     print 'step ', t
  #     # Store transition(s_t,a_t,r_t,s_{t+1}) and train the network
  #     agent.set_feedback(obs, action, reward, done)
  #     if done:
  #       print 'EPISODE: ', i, ' Steps: ', t, ' result: ', score
  #       score = 0
  #       break

if __name__ == '__main__':
  main()