import gym
import tensorflow as tf
import config
from agent import AgentAsyncDQN
import threading
from network import AsyncDQNetwork
import time

def main():

  # Set up game environments (one per thread)
  envs = [gym.make(config.EXPERIMENT) for i in range(config.NUM_THREADS)]
  action_size = envs[0].action_space.n

  network = AsyncDQNetwork(action_size)

  actor_learner_threads = [AgentAsyncDQN(args=(thread_id, envs[thread_id], network)) for thread_id in
                           range(config.NUM_THREADS)]
  for t in actor_learner_threads:
    t.start()

  last_summary_time = 0
  while True:
    if config.SHOW_TRAINING:
      for env in envs:
        env.render()

    # write summary statistics

    now = time.time()
    if now - last_summary_time > config.SUMMARY_INTERVAL:
      summary_str = network.run_summary_op()
      network.writer.add_summary(summary_str, float(T))
      last_summary_time = now

  for t in actor_learner_threads:
    t.join()

if __name__ == '__main__':
  main()