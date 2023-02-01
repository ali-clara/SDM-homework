import numpy as np
import time
from maze import Maze2D, Maze4D
from priority_queue import PriorityQueue

m = Maze2D.from_pgm('maze1.pgm')

class Node():
    def __init__(self, pos=None, parent=None, cost=None):
        self.pos = pos
        self.parent = parent
        self.cost = cost

class RRT():
    def __init__(self, start_state, goal_state, maze):
        self.start_state = start_state
        self.goal_state = goal_state
        self.maze = maze
        self.tree = []
        self.path = []

        # step size for each new node
        self.distance = 1

        start_node = Node(pos=self.start_state, cost=0)
        self.tree.append(start_node)

    def calc_distance(self, state_1, state_2):
        """Calculates Euclidian distance between two states"""
        state_1 = np.array(state_1)
        state_2 = np.array(state_2)
        return np.linalg.norm(state_1 - state_2)
    
    def sample_world(self):
        """Samples the grid-world and returns coordinate"""
        
        # samples the goal 25% of the time for goal-directed behavior
        goal_bias = 0.25
        if np.random.uniform() <= goal_bias:
            sample = (24, 24)
        else:  
            x_points = np.arange(0, 24, 1)
            y_points = np.arange(0, 24, 1)
            x_sample = np.random.choice(x_points)
            y_sample = np.random.choice(y_points)
            sample = (x_sample, y_sample)
        return sample

    def nearest_neighbor(self, point):
        """Finds the nearest vertex in the tree to 'point'
            by finding the minimium Euclidian distance between 'point' and tree nodes"""
        distance = [(self.calc_distance(node.pos, point)) for node in self.tree]
        nearest_neighbor_index = np.where(distance == min(distance))[0][0]
        nearest_neighbor = self.tree[nearest_neighbor_index].pos
        return nearest_neighbor

    ####### maybe fix this so it can only land on grid points #####
    def extend_vertex_toward_point(self, vertex, point):
        """Calculates the new coordinates found by expanding 'vertex' a set distance towards 'point'
            Returns - (x,y) tuple of new coordintes"""

        delta_x = (point[0]-vertex[0])
        delta_y = (point[1]-vertex[1])

        # account for divide by zero errors
        if delta_x == 0:
            theta = np.pi/2
        elif delta_y == 0:
            theta = 0
        else:
            theta = np.arctan(delta_y / delta_x)
        
        new_delta_y = np.sin(theta)*self.distance
        new_delta_x = np.cos(theta)*self.distance
        new_x = vertex[0] + new_delta_x
        new_y = vertex[1] + new_delta_y
        return (new_x, new_y)
        
    def goal_check(self, vertex):
        """Checks if the vertex is within 1 unit of goal
            Returns - True if vertex is, False if not"""
        x_dist = self.calc_distance(self.goal_state[0], vertex[0])
        y_dist = self.calc_distance(self.goal_state[1], vertex[1])
        if x_dist <= 1 and y_dist <= 1:
            return True
        else:
            return False

    def find_parent_node(self, vertex):
        """Finds the parent node of a given vertex in the tree
            Inputs- vertex (Node() object)
            Returns- parent (Node() object)"""
        parent_node_pos = vertex.parent

        # set cost of vertex, while we're here (cost = dist from parent to vertex)
        if parent_node_pos is not None:
            vertex.cost = self.calc_distance(parent_node_pos, vertex.pos)

        for i, node in enumerate(self.tree):
            node_position = node.pos
            threshold = 0.00001
            if parent_node_pos is None:
                return
            elif abs(self.calc_distance(parent_node_pos, node_position)) < threshold:
                parent_node_index = i
                parent_node = self.tree[parent_node_index]
                return parent_node
        
        # print(parent_node_index, parent_node_pos)
        
    def find_path(self, final_vertex):
        """Recovers path from final node
            Returns - none"""
        self.path.append(final_vertex.pos)
        parent_node = self.find_parent_node(final_vertex)
        # backtrack through the parent nodes
        while parent_node is not None:
            self.path.insert(0, parent_node.pos)
            parent_node = self.find_parent_node(parent_node)

    def calc_path_length(self):
        path_length = self.distance * len(self.path)
        return path_length
    
    def do_rrt(self):
        for _ in range(10000):
            vertex = Node()
            sample_point = self.sample_world()
            nearest_vertex = self.nearest_neighbor(sample_point)
            new_vertex = self.extend_vertex_toward_point(nearest_vertex, sample_point)
            
            # print(sample_point, nearest_vertex, new_vertex)
            if new_vertex is not None:
                if not self.maze.check_occupancy(np.round(new_vertex)):
                    vertex.pos = new_vertex
                    vertex.parent = nearest_vertex
                    self.tree.append(vertex)
                
                    if self.goal_check(new_vertex):
                        self.find_path(vertex)
                        return

        print("Expanded 10000 nodes without finding a path")
        return

if __name__ == "__main__":
    m1 = Maze2D.from_pgm('maze1.pgm')
    m2 = Maze2D.from_pgm('maze2.pgm')
    
    rrt = RRT(m1.start_state, m1.goal_state, m1)
    start = time.time()
    rrt.do_rrt()
    end = time.time()
    m1.plot_path(rrt.path, '2D, RRT \n'
                            'Path Length: '+str(np.round(rrt.calc_path_length(), 2)) +
                            ', Runtime: '+str(np.round(end-start, 4))+' sec')

    # rrt = RRT(m2.start_state, m2.goal_state, m2)
    # start = time.time()
    # rrt.do_rrt()
    # end = time.time()
    # m2.plot_path(rrt.path, '2D, RRT \n'
    #                         'Path Length: '+str(np.round(rrt.calc_path_length(), 2)) +
    #                         ', Runtime: '+str(np.round(end-start, 4))+' sec')

