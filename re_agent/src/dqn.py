from chainerrl import agents
from logging import getLogger
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc.copy_param import synchronize_parameters
from chainer import cuda
import chainer.functions as F
import chainer
import cv2
import numpy as np
import copy

class DQN(agents.double_dqn.DoubleDQN):

    def __init__(self, q_function, optimizer, replay_buffer, gamma,
                explorer, gpu=None, replay_start_size=50000,
                minibatch_size=32, update_frequency=1,
                target_update_frequency=10000, clip_delta=True,
                phi=lambda x: x,
                target_update_method='hard',
                soft_update_tau=1e-2,
                n_times_update=1, average_q_decay=0.999,
                average_loss_decay=0.99,
                batch_accumulator='mean', episodic_update=False,
                episodic_update_len=None,
                logger=getLogger(__name__),
                batch_states=batch_states,
                frame=4):
        super(DQN, self).__init__(q_function, optimizer, replay_buffer, gamma,
                explorer, gpu, replay_start_size,
                minibatch_size, update_frequency,
                target_update_frequency, clip_delta,
                phi,
                target_update_method,
                soft_update_tau,
                n_times_update, average_q_decay,
                average_loss_decay,
                batch_accumulator, episodic_update,
                episodic_update_len,
                logger,
                batch_states)
        self.frame = frame
        self.states = np.zeros((frame, 84, 84), dtype=np.uint8)

    def act(self, state):
        self.t += 1
        self.last_states = copy.deepcopy(self.states)
        self.add_states(state)
        with chainer.no_backprop_mode():
            action_value = self.model(
                self.batch_states([self.states], self.xp, self.phi), test=False)
            q = float(action_value.max.data)
            action = cuda.to_cpu(action_value.greedy_actions.data)[0]

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)
        return action

    def add_experience(self, action, reward, next_state, done):
        next_states = copy.deepcopy(self.states)
        next_states = np.roll(next_states, 1, axis=0)
        next_states[0] = self.adjust_state(copy.deepcopy(next_state))
        self.replay_buffer.append(
            state=self.states,
            action=action,
            reward=reward,
            next_state=next_states,
            next_action=0,
            is_state_terminal=done)

    def train(self):
        if self.t % self.target_update_frequency == 0:
            self.sync_target_network()
        self.replay_updater.update_if_necessary(self.t)

    def add_states(self, state):
        self.last_states = copy.deepcopy(self.states)
        self.states = np.roll(self.states, 1, axis=0)
        self.states[0] = self.adjust_state(copy.deepcopy(state))

    def stop_episode(self):
        super(DQN, self).stop_episode()
        self.states = np.zeros((self.frame, 84, 84), dtype=np.uint8)

    def adjust_state(self, state):
        gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (84, 84))
        return gray
