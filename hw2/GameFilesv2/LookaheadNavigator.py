import numpy as np
from scipy import stats
from random import randint
from networkFolder.functionList import WorldEstimatingNetwork

class Node:
    def __init__(self):
        self.position = None
        self.value = None
        self.parent = None
        self.parent_action = None

class LookaheadNavigator:
    def __init__(self):
        self.position = None
        self.spaces_visited = []
        self.nodes_expanded = []
        self.cardinal_directions = ["left", "right", "up", "down"]
        self.directions = {"left":[-1,0], "right":[1,0], "up":[0,-1], "down":[0,1]}
        self.world_estimate_net = WorldEstimatingNetwork()

    def create_world_mask(self, map):
        """Creates a mask of places explored"""
        mask = np.zeros((28, 28))
        for col in range(0, 28):
            for row in range(0, 28):
                if map[col, row] != 128:
                    mask[col, row] = 1
        return mask
    
    def estimate_world(self, map):
        """Use the neural network to predict what the world looks like"""
        mask = self.create_world_mask(map)
        world_estimate = self.world_estimate_net.runNetwork(map, mask)
        return world_estimate

    def get_neighbors(self, parent, map):
        """Gets coordinates and values of neighbors
            Parent - Node where robot currently is"""
        neighbors = []
        for action in self.directions:
            node = Node()
            node.parent = parent
            col = parent.position[1] + self.directions[action][1]
            row = parent.position[0] + self.directions[action][0]
            # if the square is off the map, throw it out
            if row > 27 or row < 0 or col < 0 or col > 27:
                continue
            # otherwise, update the value of the neighbor
            else:
                node.value = map[col, row]
                node.position = (row, col)
                node.parent_action = action

            self.nodes_expanded.append(node)
            neighbors.append(node)
        
        return neighbors

    def children_and_grandchildren(self, parent, map):
        """Gets coordinates and values of neighbors and children of neighbors
            Parent - Node where robot currently is
            Returns - lists of children and grandchildren"""
        # first, get the children of the parent node
            # also adds children to the running list of nodes we've looked at
        children = self.get_neighbors(parent, map)
        grandchildren_list = []
        # then expand each child to get the grandchildren
        for child in children:
            grandchildren = self.get_neighbors(child, map)
            for grandchild in grandchildren:
                # if the grandchild is the original parent, remove it from the nodes list
                if grandchild.position == parent.position:
                    self.nodes_expanded.remove(grandchild)
                # otherwise, add it to the current list of grandchildren
                else:
                    grandchildren_list.append(grandchild)

        return children, grandchildren_list

    def list_neighbors(self, children_list):
        """Returns the neighbor coordinates and values as a list"""
        neighbor_coords = []
        neighbor_vals = []
        for child in children_list:
            neighbor_coords.append(child.position)
            neighbor_vals.append(child.value)
        
        return neighbor_coords, neighbor_vals
    
    def get_path_values(self, grandchild):
        """Gets the map values of each node for a possible path
            Grandchild: Node"""
        map_vals = []
        map_vals.append(grandchild.value)
        # loop through the parents and append their values
        parent = grandchild.parent
        while parent is not None:
            map_vals.append(parent.value)
            parent = parent.parent

        return map_vals

    def calc_entropy(self, list):
        """Calculates the entropy of a list"""
        h = stats.entropy(list, base=2)
        return h

    def find_best_grandchild(self, grandchildren):
        """Checks all the paths to all the grandchildren and picks
            the one with the smallest entropy
            Grandchildren - a list of all possible grandchildren of the parent node"""

        # if entropy is 0, replace with a large number (otherwise throws off argmin)
        entropy_list = []
        for grandchild in grandchildren:
            path_vals = self.get_path_values(grandchild)
            h = self.calc_entropy(path_vals)
            if h == 0:
                h = 100
            entropy_list.append(h)

        h = np.array(entropy_list)
        h[h == -np.inf] = np.inf

        best_grandchild = grandchildren[h.argmin()]
        return best_grandchild, h
    
    def find_best_action(self, grandchildren):
        """Finds the actions that lead to the best grandchild"""

        best_grandchild, entropy_list = self.find_best_grandchild(grandchildren)
        actions = []
        actions.append(best_grandchild.parent_action)
        parent = best_grandchild.parent
        while parent.parent_action is not None:
            actions.append(parent.parent_action)
            parent = parent.parent

        best_action = actions[-1]
        next_node = best_grandchild.parent.position

        return best_action, next_node, entropy_list
    
    def random_action(self, robot):
        direction = None
        while direction is None:
            rands = randint(0, 3)
            direction = self.cardinal_directions[rands]
            # If it is not a valid move, reset
            if not robot.checkValidMove(direction):
                direction = None
        return direction
    
    def getAction(self, robot, map):
        """ Looks at all the neighbors, picks the direction with the lowest entropy
            (highest information gain by moving to that value)
            Returns - direction to move (str)
        """
        world = self.estimate_world(map)
        current_pos = robot.getLoc()
        # initialize the current robot location as a node
        parent = Node()
        parent.position = current_pos
        parent.value = map[current_pos[1], current_pos[0]]

        children, grandchildren = self.children_and_grandchildren(parent, world)
        best_action, next_node, entropy_list = self.find_best_action(grandchildren)
        neighbor_coords, neighbor_vals = self.list_neighbors(children)

        # if we've visited the space before, move randomly instead
        if next_node in self.spaces_visited:
            best_action = self.random_action(robot)
            print("moving randomly")
        
        # print(f"Robot at: {current_pos}, neighbors: {neighbor_coords}, moving {best_action} to {next_node}")

        self.spaces_visited.append(next_node)
        return best_action