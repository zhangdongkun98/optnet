import rllib

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam


'''
    Model Predictive Path Integral control
    ICRA-2017: Information Theoretic MPC for Model-Based Reinforcement Learning
    Ref: https://github.com/UM-ARM-Lab/pytorch_mppi.git

    Notes:
        Currently for deterministic dynamics.
'''


class MPPI(rllib.template.MethodSingleAgent):
    lr_model = 0.0003

    buffer_size = 1000000
    batch_size = 256

    start_timesteps = 30000
    
    def __init__(self, config: rllib.basic.YamlConfig, writer: rllib.basic.Writer):
        '''
            dynamics: function(state, action) -> next_state
            running_cost: function(state) -> cost
            terminal_cost: function(state) -> cost
        '''

        super().__init__(config, writer)

        self.dynamics = config.get('dynamics_cls', Dynamics)(config).to(self.device)
        self.running_cost = config.get('running_cost_cls', RunningCost)(config).to(self.device)
        self.terminal_cost = config.get('terminal_cost', lambda _: 0)

        self.actor = Actor(config, self.dynamics, self.running_cost, self.terminal_cost)

        self.optimizer = Adam(self.dynamics.parameters(), lr=self.lr_model)
        self.model_loss = nn.MSELoss()
        self._replay_buffer = config.get('buffer', rllib.td3.ReplayBuffer)(self.buffer_size, self.batch_size, self.device)
        return


    def reset(self):
        self.actor.reset()


    def update_policy(self):
        if len(self._replay_buffer) < self.start_timesteps:
            return
        super().update_policy()

        '''load data batch'''
        experience = self._replay_buffer.sample()
        state = experience.state
        action = experience.action
        next_state = experience.next_state

        loss = self.model_loss(self.dynamics(state, action), next_state)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.writer.add_scalar('loss/loss', loss.detach().item(), self.step_update)
        return


    @torch.no_grad()
    def select_action(self, state):
        '''
            state: torch.Size([1, dim_state])
        '''
        super().select_action()
        action = self.actor(state.to(self.device))
        return action





class Actor(rllib.template.Model):
    num_samples = 1000     # batch_size
    horizon = 20

    noise_sigma = 5     # control noise covariance (assume v_t ~ N(u_t, noise_sigma))

    temperature = 1.0   # positive scalar variable where larger values will allow more exploration

    def __init__(self, config, dynamics, running_cost, terminal_cost):
        super().__init__(config)

        self.dynamics = dynamics
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost

        self.noise_mean = torch.zeros((self.dim_action,), dtype=self.dtype, device=self.device)
        self.noise_sigma = torch.diag( torch.full((self.dim_action,), self.noise_sigma, dtype=self.dtype, device=self.device) )
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(self.noise_mean, covariance_matrix=self.noise_sigma)

        self.reset()
    
    
    def reset(self):
        self.actions = torch.zeros((self.horizon, self.dim_action), device=self.device)


    def forward(self, state):
        '''
            state: torch.Size([1, dim_state])
            state: torch.Size([1, dim_action])
        '''
        state = state.expand(self.num_samples, -1)  # torch.Size([num_samples, dim_state])
    
        self.actions = torch.roll(self.actions, -1, dims=0)
        self.actions[-1] = self.actions[-2]

        '''--------------------------------------------'''

        actions_u = self.actions.expand(self.num_samples, -1,-1)   # torch.Size([num_samples, horizon, dim_action])
        cost, noise = self._calculate_cost(state, actions_u)

        beta = torch.min(cost)
        cost = torch.exp(-(cost - beta) / self.temperature)
        eta = torch.sum(cost)
        omega = cost / eta

        ### option 1
        for t in range(self.horizon):
            self.actions[t] += omega.T.mm(noise[:,t,:]).squeeze()

        ### option 2, probably not right
        # self.actions += omega.T.mm(noise.view(self.num_samples, -1)).view(1, self.horizon, self.dim_action)[0]

        action = self.actions[0].view(1,self.dim_action)
        return action


    def _calculate_cost(self, state, actions_u):
        noise = self.noise_dist.sample((self.num_samples, self.horizon))   # torch.Size([num_samples, horizon, dim_action])

        actions_v = (actions_u + noise).clamp(-1,1)    # torch.Size([num_samples, horizon, dim_action])

        noise = actions_v - actions_u     # torch.Size([num_samples, horizon, dim_action])

        '''action cost'''
        actions_2e = (actions_u + 2*noise).unsqueeze(-1)
        action_cost = actions_u.unsqueeze(-2).matmul(self.noise_sigma_inv).matmul(actions_2e)
        action_cost = 0.5 * self.temperature * torch.sum(action_cost, dim=(1,2,3)).unsqueeze(-1)   # torch.Size([num_samples, 1])

        '''state cost'''
        state_cost = torch.zeros((self.num_samples, 1), dtype=self.dtype, device=self.device)   # torch.Size([num_samples, 1])
        for t in range(self.horizon-1):
            next_state = self.dynamics(state, actions_v[:,t,:])
            state_cost += self.running_cost(state)
            state = next_state
        state_cost += self.terminal_cost(state)

        return state_cost + action_cost, noise



class Dynamics(rllib.template.Model):
    def __init__(self, config):
        super().__init__(config)

        self.fc = nn.Sequential(
            nn.Linear(self.dim_state + self.dim_action, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, self.dim_state), nn.Tanh(),
        )
        self.apply(rllib.utils.init_weights)
    
    def forward(self, state, action):
        delta_state = self.fc(torch.cat([state, action], 1))
        return state + delta_state



class RunningCost(rllib.template.Model):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, state):
        '''
            state: torch.Size([num_samples, dim_state])
        '''
        raise NotImplementedError

