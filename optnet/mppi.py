import rllib


class MPPI(rllib.template.MethodSingleAgent):
    '''
        Ref: https://github.com/UM-ARM-Lab/pytorch_mppi.git
    '''

    num_samples = 1000
    horizon = 10

    temperature = 1.0  # positive scalar where larger values will allow more exploration


    def __init__(self, config, writer, dynamics, running_cost, terminal_cost=None):
        '''
            dynamics: function(state, action) -> next_state
            running_cost: function(state, action) -> cost
            terminal_cost: 
        '''

        super().__init__(config, writer)

        self.dynamics = dynamics
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost
        


    def select_action(self, state):
        super().select_action()

        state = state.to(self.device)

        

