import gym
import tensorflow as tf
import flags
from agent import AgentAsyncAC3
import threading
from network import AsyncAC3Network


FLAGS = tf.app.flags.FLAGS


def main():
    if not FLAGS.RESUME:
        if tf.gfile.Exists(FLAGS.CHECKPOINT_PATH):
            tf.gfile.DeleteRecursively(FLAGS.CHECKPOINT_PATH)
        tf.gfile.MakeDirs(FLAGS.CHECKPOINT_PATH)

        if tf.gfile.Exists(FLAGS.SUMMARY_PATH):
            tf.gfile.DeleteRecursively(FLAGS.SUMMARY_PATH)
        tf.gfile.MakeDirs(FLAGS.SUMMARY_PATH)

    # Set up game environments (one per thread)
    envs = [gym.make(FLAGS.EXPERIMENT) for _ in range(FLAGS.NUM_THREADS)]
    action_size = envs[0].action_space.n

    network = AsyncAC3Network(action_size)

    actor_learner_threads = [AgentAsyncAC3(args=(thread_id, envs[thread_id], network)) for thread_id in
                             range(FLAGS.NUM_THREADS)]
    for t in actor_learner_threads:
        t.start()


    while True:
        if FLAGS.SHOW_TRAINING:
            for env in envs:
                env.render()



    for t in actor_learner_threads:
        t.join()


if __name__ == '__main__':
    main()
