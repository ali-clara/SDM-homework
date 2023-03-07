import numpy as np
from scipy import stats
from random import randint
from networkFolder.functionList import WorldEstimatingNetwork

class GreedyNavigator:
    def __init__(self):
        self.position = None
        self.directions = {"left":[-1,0], "right":[1,0], "up":[0,-1], "down":[0,1]}
        self.neighbor_vals = {"left":None, "right":None, "up":None, "down":None}
        self.neighbor_coords = {"left":None, "right":None, "up":None, "down":None}
        self.world_estimate_net = WorldEstimatingNetwork()
        self.spaces_visited = []
        self.cardinal_directions = ["left", "right", "up", "down"]

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
        """ Looks at all the neighbors, picks the direction with the highest map value
            Returns - direction to move (str)
        """
        current_pos = robot.getLoc()
        self.spaces_visited.append(current_pos)
        self.spaces_visited.append(current_pos)
        world = self.estimate_world(map)
        self.get_neighbors(current_pos, world)

        neighbor_vals, neighbor_coords, direction_list = self.list_neightbors()

        # if all the values are the same, choose randomly
        if len(set(neighbor_vals)) == 1:
            rands = np.arange(0, len(neighbor_vals))
            best_action_index = np.random.choice(rands)
        # otherwise choose the largest
        else:
            best_action_index = np.array(neighbor_vals).argmax()
        
        best_action = direction_list[best_action_index]
        neighbor = neighbor_coords[best_action_index]

        # print(f"Robot at: {current_pos}, neighbors: {neighbor_vals}, moving {best_action}")

        # if we've visited the space before, move randomly instead
        if neighbor in self.spaces_visited:
            best_action = self.random_action(robot)
            print("moving randomly")
        
        return best_action

if __name__ == "__main__":
    gn = GreedyNavigator()
    # print(gn.directions["left"][0])