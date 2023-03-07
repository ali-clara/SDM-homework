import numpy as np
from scipy import stats
from random import randint
from networkFolder.functionList import WorldEstimatingNetwork

class EntropyNavigator:
    def __init__(self):
        self.position = None
        self.current_map_value = 0
        self.spaces_visited = []
        self.directions = {"left":[-1,0], "right":[1,0], "up":[0,-1], "down":[0,1]}
        self.neighbor_vals = {"left":None, "right":None, "up":None, "down":None}
        self.neighbor_coords = {"left":None, "right":None, "up":None, "down":None}
        self.world_estimate_net = WorldEstimatingNetwork()

    def get_neighbors(self, current_pos, map):
        """Returns the values of the neighbor positions on the map"""
        self.current_map_value = map[current_pos[0], current_pos[1]]
        for action in self.directions:
            col = current_pos[1] + self.directions[action][1]
            row = current_pos[0] + self.directions[action][0]
            # if the square is off the map, throw it out
            if row > 27 or row < 0 or col < 0 or col > 27:
                self.neighbor_vals[action] = None
            # otherwise, update the value of the neighbor
            else:
                self.neighbor_vals[action] = map[col, row]
                self.neighbor_coords[action] = (row, col)

    def list_neightbors(self):
        """Returns the dictionary of self.neighbors as a list"""
        # append all the valid directions and their values
        direction_list = []
        neighbor_vals = []
        neighbor_coords = []
        for key in self.neighbor_vals.keys():
            if self.neighbor_vals[key] is not None:
                direction_list.append(key)
                neighbor_vals.append(self.neighbor_vals[key])
                neighbor_coords.append(self.neighbor_coords[key])
        
        return neighbor_vals, neighbor_coords, direction_list
    
    def calc_entropy(self, neighbor_vals):
        """calculates the information gained adding each neighbor"""
        # if the current pixel val is 0, replace with a small number
            # (otherwise drives all entropy vals to 0)
        if self.current_map_value == 0:
            self.current_map_value = 0.001

        # find the entropy of each neighbor
        # if entropy is 0, replace with a large number (for doing argmins later)
        entropy_list = []
        for neighbor in neighbor_vals:
            h = stats.entropy([self.current_map_value, neighbor], base=2)
            if h == 0:
                h = 100
            entropy_list.append(h)
        return entropy_list
    
    def find_best_action(self, pixel_list, direction_list, random=False):
        # if all entropy values are the same, choose randomly
        if len(set(pixel_list)) == 1 or random == True:
            rands = np.arange(0, len(pixel_list))
            best_action_index = np.random.choice(rands)
        # otherwise choose the smallest
        else:
            best_action_index = np.array(pixel_list).argmin()
        
        best_action = direction_list[best_action_index]
        return best_action, best_action_index

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
    
    def getAction(self, robot, map):
        """ Looks at all the neighbors, picks the direction with the lowest entropy
            (highest information gain by moving to that value)
            Returns - direction to move (str)
        """
        current_pos = robot.getLoc()
        world = self.estimate_world(map)
        self.get_neighbors(current_pos, world)
        neighbor_vals, neighbor_coords, direction_list = self.list_neightbors()
        entropy_list = self.calc_entropy(neighbor_vals)

        best_action, best_action_index = self.find_best_action(entropy_list, direction_list)
        neighbor = neighbor_coords[best_action_index]

        # if we've visited the space before, move randomly instead
        if neighbor in self.spaces_visited:
            best_action, best_action_index = self.find_best_action(entropy_list, direction_list, random=True)

        # print(f"Robot at: {current_pos}, neighbors: {neighbor_coords}, moving {best_action} to {neighbor}")

        self.spaces_visited.append(neighbor)
        return best_action

if __name__ == "__main__":
    en = EntropyNavigator()

    base = 2  # work in units of bits
    pk = np.array([254.0, 242.0])
    # print(stats.entropy(pk, base=base))

    neighbor_list = [0, 0, 0, 0]
    entropy_array = []
    for neighbor in neighbor_list:
        h = stats.entropy([200, neighbor], base=2)
        if h == 0:
            h = 100
        entropy_array.append(h)

    entropy_array = np.array(entropy_array)
    print(entropy_array)
    print(entropy_array.argmin())

    # if current_val is 0, current_val = 0.0001 (otherwise makes all vals 0)
    # find the entropy of adding each neighbor
    # if all values are the same, move randomly
    # otherwise, move in the direction of the smallest entropy value