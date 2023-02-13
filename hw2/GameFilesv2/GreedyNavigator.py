__author__ = 'Caleytown'
import numpy as np
from scipy import stats
from random import randint

"""
robot is at location and can see all squares around it
for possible action:
    look at square
    create map, mask
    calculate entropy of map
    determine which action has smallest entropy
    robot action = action
    
"""

class GreedyNavigator:
    def __init__(self):
        self.position = None
        self.directions = {"left":[-1,0], "right":[1,0], "up":[0,1], "down":[0,-1]}
        self.neighbor_vals = {"left":None, "right":None, "up":None, "down":None}

    def get_neighbors(self, current_pos, map):
        """Returns the values of the neighbor positions on the map"""
        neighbor_vals = []
        for action in self.directions:
            col = current_pos[1] + self.directions[action][0]
            row = current_pos[0] + self.directions[action][1]
            # if the square is off the map, throw it out
            if row > 27 or row < 0 or col < 0 or col > 27:
                self.neighbor_vals[action] = None
            # otherwise, update the value of the neighbor
            else:
                self.neighbor_vals[action] = map[col, row]
        
    def getAction(self, robot, map):
        """ Looks at all the neighbors, picks the direction with the highest map value
            Returns - direction to move (str)
        """
        current_pos = robot.getLoc()
        self.get_neighbors(current_pos, map)

        # append all the valid directions and their values
        direction_list = []
        neighbor_list = []
        for key in self.neighbor_vals.keys():
            if self.neighbor_vals[key] is not None:
                direction_list.append(key)
                neighbor_list.append(self.neighbor_vals[key])

        # if all the values are the same, choose randomly
        if len(set(neighbor_list)) == 1:
            rands = np.arange(0, len(neighbor_list))
            best_action_index = np.random.choice(rands)
        # otherwise choose the largest
        else:
            best_action_index = np.array(neighbor_list).argmax()
        
        best_action = direction_list[best_action_index]
        return best_action

if __name__ == "__main__":
    gn = GreedyNavigator()
    # print(gn.directions["left"][0])