__author__ = 'Caleytown'

import gzip
import numpy as np
from scipy import stats
from PIL import Image
from RobotClass import Robot
from GameClass import Game
from RandomNavigator import RandomNavigator
from GreedyNavigator import GreedyNavigator
from networkFolder.functionList import Map, WorldEstimatingNetwork, DigitClassificationNetwork

# Create a Map Class Object
map = Map()

# Get the current map from the Map Class
data = map.map

# print(data)

# Print the number of the current map
print(map.number)

# Create a Robot that starts at (0,0)
# The Robot Class stores the current position of the robot
# and provides ways to move the robot 
robot = Robot(0, 0)

# The RandomNavigator class makes the robot move in random directions
# TODO: You will want to write other navigators with different exploration strategies.
# navigator = RandomNavigator()
navigator = GreedyNavigator()

# Create a Game object, providing it with the map data, the goal location of the map, the navigator, and the robot
game = Game(data, map.number, navigator, robot)

# This loop runs the game for 1000 ticks, stopping if a goal is found.
for x in range(0, 1000):
    found_goal = game.tick()
    # print(f"{game.getIteration()}: Robot at: {robot.getLoc()}, Score = {game.getScore()}")
    # if found_goal:
    #     print(f"Found goal at time step: {game.getIteration()}!")
    #     break
print(f"Final Score: {game.score}")

# Show how much of the world has been explored
im = Image.fromarray(np.uint8(game.exploredMap)).show()

# Create the world estimating network
uNet = WorldEstimatingNetwork()

# Create the digit classification network
classNet = DigitClassificationNetwork()

# This loop shows how you can create a mask, an grid of 0s and 1s
# where 0s represent unexplored areas and 1s represent explored areas
# This mask is used by the world estimating network
mask = np.zeros((28, 28))
for col in range(0, 28):
    for row in range(0, 28):
        if game.exploredMap[col, row] != 128:
            mask[col, row] = 1

# Creates an estimate of what the world looks like
image = uNet.runNetwork(game.exploredMap, mask)

# Show the image of the estimated world
Image.fromarray(image).show()

# Use the classification network on the estimated image
# to get a guess of what "world" we are in (e.g., what the MNIST digit of the world)
char = classNet.runNetwork(image)
# get the most likely digit 
# print(char[0])
estimate_array = np.abs(np.array(char[0]))
smallest = np.min(estimate_array)
baseline_sub = estimate_array - smallest
baseline_nonzero_mask = np.nonzero(baseline_sub)
baseline_nonzero = baseline_sub[baseline_nonzero_mask]

print(estimate_array)
print(baseline_sub)
print(baseline_nonzero)
if all(np.abs(baseline_nonzero) > 1.5):
    print("confident in this answer")

print(char.argmax())
# print(char.argmin())

# print(stats.entropy(char[0]))
# Get the next map to test your algorithm on.
map.getNewMap()


