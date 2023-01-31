import numpy as np
import time
from maze import Maze2D, Maze4D
from priority_queue import PriorityQueue

class Node():
    def __init__(self, pos=None, parent=None, cost=0):
        self.pos = pos
        self.parent = parent
        # cost to come, distance from start -> node
        self.cost = cost

class AStar():
    def __init__(self, start_state, goal_state, maze):
        self.start_state = start_state
        self.goal_state = goal_state
        self.maze = maze
        self.path = []
        self.graph = []

        self.open_list = PriorityQueue()
        self.closed_list = PriorityQueue()

        start_node = Node(pos=self.start_state, cost=0)
        self.open_list.insert(start_node.pos, start_node.cost)
        self.graph.append(start_node)
    
    def calc_distance(self, state_1, state_2):
        """Calculates Euclidian distance between two states"""
        state_1 = np.array(state_1)
        state_2 = np.array(state_2)
        return np.linalg.norm(state_1 - state_2)
    
    def euclidian_heuristic(self, current_state, use_epsilon=False, epsilon=None):
        """Calculates h(n), the heuristic cost-to-go 
            Inputs - Current state (tuple, array, list), Goal state (tuple, array, list)
            Returns - Euclidian distance between state and goal"""
        cost_to_go = self.calc_distance(self.goal_state, current_state)
        if use_epsilon == True:
            return cost_to_go * epsilon
        else:
            return cost_to_go

    def get_neighbors(self, current_state):
        """Gets neighbors of current_state not in closed list
            Returns - neighbor (x,y) tuple"""
        neighbors = []
        current_index = self.maze.index_from_state(current_state)
        x = [self.maze.state_from_index(pos) for pos in self.maze.get_neighbors(current_index)]
        for neighbor in x:
            if not self.closed_list.test(neighbor):
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
            node_position = node.pos
            if node_position == n_best:
                node_index = i
                return self.graph[node_index]
        
    def calc_path_length(self):
        return self.graph[-1].cost

    def do_astar(self, use_epsilon=False, epsilon=None):
        """Runs the A* algorithm and saves the final path in self.path
            Returns - none"""
        while len(self.open_list) != 0:
            # find loweset priority node and remove from open list
            n_best = self.get_nbest()

            # add n_best state information to closed list
            self.closed_list.insert(n_best.pos, n_best.cost)
            self.graph.append(n_best)

            # if we're at the goal, we're done
            if n_best.pos == self.goal_state:
                self.find_path(n_best)
            # otherwise, expand neighbors and either add to or update the closed list
            else:
                neighbors = self.get_neighbors(n_best.pos)
                for n in neighbors:
                    # make node out of neighbor position
                    neighbor = Node(pos=n)

                    # find cost-to-come and cost-to-go
                    dist_from_nbest = self.calc_distance(neighbor.pos, n_best.pos)
                    cost_to_come = n_best.cost + dist_from_nbest 
                    heuristic = self.euclidian_heuristic(neighbor.pos, use_epsilon=use_epsilon, epsilon=epsilon)
                    total_cost = cost_to_come + heuristic

                    # if it's not in the open list, add it
                    if not self.open_list.test(neighbor):
                        self.open_list.insert(neighbor.pos, total_cost)
                        neighbor.parent = n_best
                        neighbor.cost = cost_to_come 
                        self.graph.append(neighbor)

                    # if it is in the open list but has a higher cost, update it
                    elif (cost_to_come < neighbor.cost):
                        self.open_list.insert(neighbor.pos, total_cost)
                        neighbor.parent = n_best
                        neighbor.cost = cost_to_come 
                        self.graph.append(neighbor)

if __name__ == "__main__":
    
    m1 = Maze2D.from_pgm('maze1.pgm')
    m2 = Maze2D.from_pgm('maze2.pgm')

    def go(maze, title):
        astar = AStar(maze.start_state, maze.goal_state, maze)
        start = time.time()
        astar.do_astar()
        end = time.time()
        maze.plot_path(astar.path, title+', 2D, A* \n'
                                'Path Length: '+str(np.round(astar.calc_path_length(),2)) +
                                ', Runtime: '+str(np.round(end-start, 4))+' sec')

    # go(m1, "Maze 1")
    # go(m2, "Maze 2")


    def greedy_astar(maze):
        astar = AStar(maze.start_state, maze.goal_state, maze)
        epsilon = 1000
        while epsilon != 1:
            start = time.time()
            astar.do_astar(use_epsilon=True, epsilon=epsilon)
            end = time.time()
            astar.maze.plot_path(astar.path, 'Epsilon: '+str(epsilon)+', 2D, A* \n'
                                'Path Length: '+str(np.round(astar.calc_path_length(),2)) +
                                ', Nodes Expanded: '+str(len(astar.closed_list)))
            epsilon = epsilon - 0.5*(epsilon - 1)
            if epsilon < 1.001:
                epsilon = 1

    greedy_astar(m1)




