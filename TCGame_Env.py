from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product



class TicTacToe():
    def __init__(self):
        """initialise the board"""
        
        # initialise state as an array
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array or matrix
        self.winning_checks = [(0,1,2),(3,4,5),(6,7,8), ## all rows
                               (0,3,6),(1,4,7),(2,5,8), ## all columns
                               (0,4,8),(2,4,6)] ## two diagonals
        self.reset()

    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
        
        ## for each possible check , we will see if the sum is 15
        for (i,j,k) in self.winning_checks: 
            if (np.nan not in (i,j,k)) and (curr_state[i]+curr_state[j]+curr_state[k] == 15):
                return True
        return False

    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) ==0:
            return True, 'Tie'

        else:
            return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)


    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = list(product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0]))
        env_actions = list(product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1]))
        return (agent_actions, env_actions)



    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        out_state = curr_state.copy()
        out_state[curr_action[0]] = curr_action[1] ##add the used number
        return out_state

    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
        
        ##get output state after agent's move
        out_state = self.state_transition(curr_state,curr_action)
        
        ##check if it's a terminal
        terminal_bool , terminal_value = self.is_terminal(out_state)
        
        if terminal_bool:
            if terminal_value == "Win":  ##if win , return 10 
                return (out_state,10,terminal_bool)
            elif terminal_value == "Tie": ##if tie, return 0
                return (out_state,0,terminal_bool)
        else:
            reward = -1
            
            ##play the environment
            agent_actions, env_actions = self.action_space(out_state)
            env_random_action = random.choice(env_actions)
            out_state = self.state_transition(out_state, env_random_action)

            terminal_bool, terminal_value = self.is_terminal(out_state)

            # check for env win
            if (terminal_value == 'Win'):
                reward = -10
            elif (terminal_value == 'Tie'):
                reward = 0
            return out_state, reward, terminal_bool


    def reset(self):
        return self.state
