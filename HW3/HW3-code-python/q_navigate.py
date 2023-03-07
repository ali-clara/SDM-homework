from maze import Maze
import time
import numpy as np

class ValueIteration:
    def __init__(self, maze, maze_filename, discount_factor, noise):
        # maze init
        self.maze = maze
        maze.load_maze(maze_filename)
        maze.reset()
        self.maze_shape = maze.rewards.shape
        
        # val iteration init
        self.discount = discount_factor
        self.noise = noise
        self.num_actions = len(maze.actions)
        self.states = self.init_states()
        self.q_values = np.zeros((self.maze_shape[0], self.maze_shape[1]))
        self.policy = None

    def init_states(self):
        states = np.empty((self.maze_shape[0], self.maze_shape[1]), dtype=tuple)
        for i in range(self.maze_shape[0]):
            for j in range(self.maze_shape[1]):
                states[i,j] = (i,j)
        
        return states
    
    def reward_function(self, position=None):
        """Position in (cols, rows)"""
        return self.maze.get_reward(position)
    
    def transition_probability(self, state, action):
        """Finds the probability that you get to each state
            based on the current state and the action
            Returns - p: array of probs corresponding to each action"""
        p_wrong_move = 0.1/3
        p_right_move = 0.9

        p = np.array([p_wrong_move]*4)
        p[action] = p_right_move

        for action in self.maze.actions:
            if self.maze.is_move_valid(action, state) is None:
                p[action] = 0

        return p

    def q_vals_of_neighbors(self, state):
        q_vals = []
        for action in self.maze.actions:
            neighbor = self.maze.move_maze(action, state)
            if self.maze.is_move_valid(action, state) is None:
               q_val = 0
            else:
                q_val = self.q_values[neighbor]

            q_vals.append(q_val)

        return q_vals

    def value_iteration(self, tol = 1e-3):
        for i in range(50):
            delta = 0
            # for each state
            for state in self.states.flatten():
                # grab previous state value and initialize the list of new q vals
                prev_q = self.q_values[state]
                q_list = np.zeros(self.num_actions)
                # for each action
                for action in self.maze.actions:
                    # find the probability of visiting a state given an action and
                        # multiply it by the value of each state 
                    probability = self.transition_probability(state, action)
                    # print(probability)
                    next_state_q_val = self.q_vals_of_neighbors(state)
                    utility_update = probability * next_state_q_val
                    # print(utility_update)
                    # perform bellman's update
                    q_list[action] = self.reward_function(state) + self.discount*np.sum(utility_update)

                # update the state value
                q_update = max(q_list)
                self.q_values[state] = q_update
                # update stopping criteria
                delta = max(delta, np.abs(prev_q - q_update))

            if delta < tol:
                print(f"Change in Q value between iterations was {delta} after {i} iterations")
                break

    def best_action(self, state):
        neighbor_vals = self.q_vals_of_neighbors(state)
        best_action = np.argmax(neighbor_vals)

        return best_action
    
    def run(self):
        self.maze.draw_maze(self.q_values.round(3))
        while not self.maze.get_observation():
            self.value_iteration()
            maze.step(self.best_action(self.maze.position))
            self.maze.draw_maze(self.q_values.round(3))
            time.sleep(0.1)

    def visualize(self):
        self.maze.draw_maze(self.q_values.round(3))
        time.sleep(10)

if __name__ == "__main__":
    noise = 0.1
    discount = 0.9
    maze_filename = "mazes/maze0.txt"
    maze = Maze(noise=noise, discount=discount)
    vi = ValueIteration(maze, maze_filename, discount, noise)

    # print(vi.states.flatten())
    # print(vi.maze_shape)
    # print(vi.reward_function((5,3)))
    # print(vi.values)

    # for state in vi.states.flatten():
    #     print(state)
    #     print(vi.values[state])

    vi.run()
    