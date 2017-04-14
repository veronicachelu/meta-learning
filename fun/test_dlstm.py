import tensorflow as tf
from tensorflow import map_fn
import numpy as np
import tensorflow.contrib.layers as layers
from tensorflow.contrib import rnn


def conditional_backprop(do_backprop, tensor):
    do_backprop = tf.Print(do_backprop, [do_backprop], "switch query")
    t = tf.cond(tf.cast(do_backprop, tf.bool),
                lambda: tf.Print(tensor, [0],
                                 "backprop enabled for " + tensor.op.name),
                lambda: tf.zeros_like(tensor))
    y = t + tf.stop_gradient(tensor - t)
    return y


with tf.Session() as sess:
    def fast_dlstm(s_t, state_in):
        def dilate_one_time_step(one_h, switcher, num_chunks):
            h_slices = []
            h_size = 256
            chunk_step_size = h_size // num_chunks
            for switch_step, h_step in zip(range(num_chunks), range(0, h_size, chunk_step_size)):
                one_switch = switcher[switch_step]
                h_s = conditional_backprop(one_switch, one_h[h_step: h_step + chunk_step_size])
                h_slices.append(h_s)
            dh = tf.stack(h_slices)
            dh = tf.reshape(dh, [-1, 256])
            return dh

        lstm = rnn.LSTMCell(256, state_is_tuple=True)
        chunks = 8

        def dlstm_scan_fn(previous_output, current_input):
            out, state_out = lstm(current_input, previous_output[1])
            i = previous_output[2]
            basis_i = tf.one_hot(i, depth=chunks)
            state_out_dilated = dilate_one_time_step(tf.squeeze(state_out[0]), basis_i, chunks)
            state_out = rnn.LSTMStateTuple(state_out_dilated, state_out[1])
            i += tf.constant(1)
            new_i = tf.mod(i, chunks)
            return out, state_out, new_i

        rnn_outputs, final_states, mod_idxs = tf.scan(dlstm_scan_fn,
                                                      tf.transpose(s_t, [1, 0, 2]),
                                                      initializer=(
                                                      state_in[1], rnn.LSTMStateTuple(*state_in), tf.constant(0)))

        state_out = [final_states[0][-1, 0, :], final_states[1][-1, 0, :]]
        cell_states = final_states[0][:, 0, :]
        out_states = final_states[1][:, 0, :]
        return out_states, cell_states, state_out


    x_t = tf.placeholder(dtype=tf.float32, shape=[1, 128, 256])
    state_in = [tf.placeholder(shape=(1, 256), dtype='float32'),
                tf.placeholder(shape=(1, 256), dtype='float32')]
    dlstm_out_states, dlstm_cell_states, dlstm_current_state = fast_dlstm(x_t, state_in=state_in)
    print(dlstm_out_states)
    print(dlstm_cell_states)
    print(dlstm_current_state)