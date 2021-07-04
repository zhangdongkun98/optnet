

import rllib
import optnet

import time
import numpy as np

import gym
import torch

from rllib.args import generate_args


def dynamics(state, action):
    # true dynamics from gym

    th = state[:, 0].view(-1, 1)
    thdot = state[:, 1].view(-1, 1)

    g = 10
    m = 1
    l = 1
    dt = 0.05

    u = action
    u = torch.clamp(u, -2, 2)    # TODO


    newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
    newth = th + newthdot * dt
    newthdot = torch.clamp(newthdot, -8, 8)

    next_state = torch.cat((newth, newthdot), dim=1)
    return next_state


def running_cost(state):
    theta = state[:, 0]
    theta_dt = state[:, 1]
    cost = angle_normalize(theta) ** 2 + 0.1 * theta_dt ** 2
    return cost.unsqueeze(1)

def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)



def main():
    seed = 1998
    rllib.basic.setup_seed(seed)

    ############## Hyperparameters ##############

    render = False
    render = True
    max_episodes = 10000        # max training episodes
    
    config = rllib.basic.YamlConfig({}, 'None')
    args = generate_args()
    config.update(args)

    env_name = "Pendulum-v0"
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    setattr(env, 'dim_state', 2)       # TODO
    setattr(env, 'dim_action', env.action_space.shape[0])

    Method = optnet.mppi.MPPI

    config.set('dim_state', env.dim_state)
    config.set('dim_action', env.dim_action)
    config.set('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    model_name = Method.__name__ + '-' + env_name
    writer = rllib.basic.create_dir(config, model_name)
    method = Method(config, writer, dynamics=dynamics, running_cost=running_cost)

    #############################################

    for i_episode in range(max_episodes):
        running_reward = 0
        avg_length = 0
        env.reset()
        method.reset()
        state = env.state    # TODO
        while True:

            # t1 = time.time()
            action = method.select_action( torch.from_numpy(state).unsqueeze(0).float() )
            # t2 = time.time()
            # print('time: ', t2-t1)
            # time.sleep(0.4)
            
            _, reward, done, _ = env.step(action.cpu().numpy().flatten())
            next_state = env.state

            # experience = rllib.template.Experience(
            #         state=torch.from_numpy(state).float().unsqueeze(0),
            #         next_state=torch.from_numpy(next_state).float().unsqueeze(0),
            #         action=action.cpu(), reward=reward, done=done)
            # method.store(experience)

            state = next_state

            running_reward += reward
            avg_length += 1
            if render: env.render()
            if done: break
        
        ### logging
        print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
        writer.add_scalar('index/reward', running_reward, i_episode)
        writer.add_scalar('index/avg_length', avg_length, i_episode)


            
if __name__ == '__main__':
    main()
    
