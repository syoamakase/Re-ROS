## DQN
Double DQN implementation with chainerrl

### dependencies
- chainer
- chainerrl

### usage
```py
from chainer import links as L
from chainer import optimizers
from chainerrl import links
from chainerrl import replay_buffer
from chainerrl.action_value import DiscreteActionValue
from dqn import DQN
from network import NatureDQNHead

n_act = 4
n_frame = 4

# q function
q_func = links.Sequence(
    NatureDQNHead(n_frame),
    L.Linear(512, n_act),
    DiscreteActionValue
)

# optimizer
opt = optimizers.RMSpropGraves(
    lr=2.5e-4, alpha=0.95, momentum=0.95, eps=1e-2)
opt.setup(q_func)

# Replay Buffer
rbuf = replay_buffer.ReplayBuffer(10 ** 5)

# DQN agent
# set None at gpu, if use CPU
agent = DQN(q_func, opt, rbuf, gpu=0, gamma=0.99,
    explorer=None, replay_start_size=10 ** 4,
    target_update_frequency=10 ** 4,
    update_frequency=n_frame, frame=n_frame)
```

### reference
```py
# get greedy action from state
# state should be RGB colored image
action = agent.act(state)

# add experience to replay buffer
# next_state should be same shape as state above
# done is boolean value
agent.add_experience(action, reward, next_state, done)

# train agent, using experiences stored in replay buffer
# updating frequency dependes on update_frequency
agent.train()

# refresh states at the end of episode
agent.stop_episode()

# save model
agent.save(dir_name)

# load model
agent.load(dir_name)
```
