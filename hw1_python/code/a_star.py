import numpy as np
import pandas as pd
import time
from maze import Maze2D, Maze4D
from priority_queue import PriorityQueue

class Node():
    def __init__(self, pos=None, parent=None, cost=0):
        self.name = str(pos)
        self.pos = pos
        self.parent = parent
        # cost to come, distance from start -> node
        self.cost = cost

class AStar():
    def __init__(self, start_state, goal_state, maze, dimension):
        self.start_state = start_state
        self.goal_state = goal_state
        self.maze = maze
        self.path = []
        self.graph = []

        self.open_list = PriorityQueue()
        self.closed_list = PriorityQueue()

        self.maze_dimension = dimension
        self.nodes_expanded = 0
        self.path_length = 0

        start_node = Node(pos=self.start_state, cost=0)
        self.open_list.insert(start_node.name, start_node.cost)
        self.graph.append(start_node)
    
    def calc_distance(self, state_1, state_2):
        """Calculates Euclidian distance between two states"""
        state_1 = np.array(state_1)
        state_2 = np.array(state_2)
        return np.linalg.norm(state_1 - state_2)

    def calc_cost_to_come(self, neighbor, n_best):
        """Calculates g(n)"""
        dist_from_nbest = self.calc_distance(neighbor.pos, n_best.pos)
        if self.maze_dimension == "2D":
            cost_to_come = n_best.cost + dist_from_nbest 
            return cost_to_come
        elif self.maze_dimension == "4D":
            velocity = np.sqrt(neighbor.pos[2]**2 + neighbor.pos[3]**2)
            if velocity == 0:
                cost_to_come = n_best.cost
            else:
                cost_to_come = n_best.cost + dist_from_nbest / velocity
            return cost_to_come

    def euclidian_heuristic(self, current_state, use_epsilon=False, epsilon=None):
        """Calculates h(n), the heuristic cost-to-go 
            Inputs - Current state (tuple, array, list), Goal state (tuple, array, list)
            Returns - Euclidian distance between state and goal"""
        cost_to_go = self.calc_distance(self.goal_state, current_state)
        if use_epsilon == True:
            return cost_to_go * epsilon
        else:
            return cost_to_go

    def time_heuristic(self, current_state, use_epsilon=False, epsilon=None):
        max_velocity = 2
        cost_to_go = self.calc_distance(self.goal_state, current_state) / max_velocity
        if use_epsilon == True:
            return cost_to_go * epsilon
        else:
            return cost_to_go
    
    def get_neighbors(self, current_state):
        """Gets neighbors of current_state not in closed list
            Returns - neighbor (tuple (2D) or array (4D))"""
        neighbors = []
        # current_state = [current_state[0][0], current_state]
        current_index = self.maze.index_from_state(current_state)
        x = [self.maze.state_from_index(pos) for pos in self.maze.get_neighbors(current_index)]
        for neighbor in x:
            # closed list contains strings of states to represent the nodes
            if not self.closed_list.test(str(neighbor)):
                neighbors.append(neighbor)
        return neighbors

    def find_path(self, final_node):
        """Recovers path from final node
            Returns - none"""
        self.path.append(final_node.pos)
        parent_node = final_node.parent
        # backtrack through the parent nodes
        while parent_node is not None:
            self.path.insert(0, parent_node.pos)
            parent_node = parent_node.parent
    
    def get_nbest(self):
        """Removes n_best from open list 
            Returns - n_best (lowest cost node)"""
        # lowest priority state
        n_best = self.open_list.pop()

        # find node that matches that state
        for i, node in enumerate(self.graph):
            node_name = node.name
            if node_name == n_best:
                node_index = i
                return self.graph[node_index]
        
    def calc_path_length(self):
        """Calculates the length of the path traveled"""
        prev_node_pos = (0,0)
        path_length = 0
        for node in self.path:
            if node is not None:
                if self.maze_dimension == "2D":
                    node_pos = node
                elif self.maze_dimension == "4D":
                    node_pos = node[:2]
            distance_traveled = self.calc_distance(node_pos, prev_node_pos)
            path_length += distance_traveled
            prev_node_pos = node_pos

        self.path_length = path_length
        return path_length
    
    def do_astar(self, use_epsilon=False, epsilon=None, runtime=100):
        """Runs the A* algorithm and saves the final path in self.path
            Returns - none"""
        start = time.time()
        iter = 0
        while len(self.open_list) != 0:
            if time.time() - start > runtime:
                print(f'Exceeded runtime of {runtime} sec')
                break

            iter += 1
            # find loweset priority node and remove from open list
            n_best = self.get_nbest()

            # add n_best state information to closed list
            self.closed_list.insert(n_best.name, n_best.cost)
            self.graph.append(n_best)

            # if we're at the goal, we're done
            if np.array_equal(n_best.pos, self.goal_state):
                self.find_path(n_best)
            # otherwise, expand neighbors and either add to or update the closed list
            else:
                neighbors = self.get_neighbors(n_best.pos)
                for n in neighbors:
                    # make node out of neighbor position
                    neighbor = Node(pos=n)

                    # find cost-to-come and cost-to-go
                    cost_to_come = self.calc_cost_to_come(neighbor, n_best)
                    if self.maze_dimension == "2D":
                        heuristic = self.euclidian_heuristic(neighbor.pos, use_epsilon=use_epsilon, epsilon=epsilon)
                    elif self.maze_dimension == "4D":
                        heuristic = self.time_heuristic(neighbor.pos, use_epsilon=use_epsilon, epsilon=epsilon)
                    total_cost = cost_to_come + heuristic

                    # if it's not in the open list, add it
                    if not self.open_list.test(neighbor.name):
                        self.open_list.insert(neighbor.name, total_cost)
                        neighbor.parent = n_best
                        neighbor.cost = cost_to_come 
                        self.graph.append(neighbor)

                    # if it is in the open list but has a higher cost, update it
                    elif (cost_to_come < neighbor.cost):
                        self.open_list.insert(neighbor.name, total_cost)
                        neighbor.parent = n_best
                        neighbor.cost = cost_to_come 
                        self.graph.append(neighbor)

        self.nodes_expanded = iter

def tabulate_data(epsilon):

        # initialize dataframe columns and first row
        initialize = [{"initialize": 0}]
        df = pd.DataFrame(initialize)
        df.insert(0, "Nodes", 0)
        df.insert(0, "Path", 0)
        df.drop(columns=["initialize"], inplace=True)
        df.insert(0, "Epsilon", epsilon)
        df.drop([0], axis=0, inplace=True)

        return df

if __name__ == "__main__":
    
    m1 = Maze2D.from_pgm('maze1.pgm')
    m2 = Maze2D.from_pgm('maze2.pgm')

    m1_4d = Maze4D.from_pgm('maze1.pgm')
    m2_4d = Maze4D.from_pgm('maze2.pgm')

    def run_astar(maze, dimension):
        astar = AStar(maze.start_state, maze.goal_state, maze, dimension)
        start = time.time()
        astar.do_astar()
        end = time.time()
        maze.plot_path(astar.path, dimension+', A* \n'
                                'Path Length: '+str(np.round(astar.calc_path_length(),2)) +
                                ', Runtime: '+str(np.round(end-start, 4))+' sec')
        print(astar.nodes_expanded)

    def run_greedy_astar(maze, dimension):
        runtime = 1000
        epsilon = 10
        df = tabulate_data(epsilon)
        
        while epsilon != 1:
            astar = AStar(maze.start_state, maze.goal_state, maze, dimension)
            astar.do_astar(use_epsilon=True, epsilon=epsilon, runtime=runtime)
            astar.maze.plot_path(astar.path, dimension+ ', Epsilon: '+str(epsilon)+', A* \n'
                                'Path Length: '+str(np.round(astar.calc_path_length(),2)) +
                                ', Nodes Expanded: '+str(astar.nodes_expanded))

            # ensure we have up-to-date path length
            astar.calc_path_length()
            # add info to dataframe
            df.loc[len(df.index)] = [epsilon, astar.path_length, astar.nodes_expanded]

            epsilon = epsilon - 0.5*(epsilon - 1)
            if epsilon < 1.001:
                epsilon = 1

        print(f"Runtime {runtime} sec")
        print(df)

        

    # run_astar(m1, "2D")
    # run_astar(m2, "2D")
    # run_astar(m1_4d, "4D")
    run_astar(m2_4d, "4D")

    # run_greedy_astar(m2_4d, "4D")




