__author__ = 'Caleytown'

import time
import numpy as np
from PIL import Image
from RobotClass import Robot
from GameClass import Game
from GreedyNavigator import GreedyNavigator
from GreedyEntropyNavigator import EntropyNavigator
from LookaheadNavigator import LookaheadNavigator
from networkFolder.functionList import Map, WorldEstimatingNetwork, DigitClassificationNetwork

class Solver():

    def __init__(self, navigator, map):
        # Print the number of the current map
        self.ground_truth = map.number
        print(f"Ground truth: {self.ground_truth}")
        # Get the current map from the Map Class
        data = map.map
        # Create a Robot at a start position
        self.robot = Robot(0, 0)
        # Create a Game object, providing it with the map data, the goal location of the map, the navigator, and the robot
        self.game = Game(data, map.number, navigator, self.robot)
        # initialize networks
        self.world_estimate_net = WorldEstimatingNetwork()
        self.digit_estimate_net = DigitClassificationNetwork()

    def create_world_mask(self):
        mask = np.zeros((28, 28))
        for col in range(0, 28):
            for row in range(0, 28):
                if self.game.exploredMap[col, row] != 128:
                    mask[col, row] = 1
        return mask
    
    def ready_to_guess(self, digit_estimate):
        # take the smallest absolute value, subtract from all other values in the list
        # if all elements of the new list are larger than a threshold, we're confident
        digit_estimate = np.abs(np.array(digit_estimate))
        smallest = np.min(digit_estimate)
        baseline = digit_estimate - smallest
        baseline_nonzero_mask = np.nonzero(baseline)
        baseline_nonzero = baseline[baseline_nonzero_mask]

        threshold = 3.5
        if all(np.abs(baseline_nonzero) > threshold):
            return True
        else:
            return False

    def make_guess(self, digit_estimate):
        return digit_estimate.argmax()

    def find_goal(self, guess):
        """returns the (x,y) position of the goal and [x_action, y_action] to take"""
        if guess in [0, 1, 2]:
            return (0, 27), ["left", "down"]
        if guess in [3, 4, 5]:
            return (27, 27), ["right", "down"]
        if guess in [6, 7, 8, 9]:
            return (27, 0), ["right", "up"]

    def find_distance_to_goal(self, goal):
        """goal is a tuple of the goal position
            Returns tuple (x_distance, y_distance)"""
        x = self.robot.xLoc
        y = self.robot.yLoc
        x_abs_dist = np.abs(x-goal[0])
        y_abs_dist = np.abs(y-goal[1])

        return x_abs_dist, y_abs_dist

    def move_to_goal(self, goal_pos, goal_actions):
        """moves the robot to the corner of the map 
            representing the goal for the current digit estimate"""
        x_dist_to_goal, y_dist_to_goal = self.find_distance_to_goal(goal_pos)
        
        for x in range(x_dist_to_goal):
            self.game.tick(action=goal_actions[0])
            print(f"{self.game.getIteration()}: Robot at: {self.robot.getLoc()}, Score = {self.game.getScore()}")
        for y in range(y_dist_to_goal):
            self.game.tick(action=goal_actions[1])
            print(f"{self.game.getIteration()}: Robot at: {self.robot.getLoc()}, Score = {self.game.getScore()}")
        
    def check_goal(self, guess):
        # Find the goal for the current estimate and move there
        goal_pos, goal_actions = self.find_goal(guess)
        print(f"Thinks the goal is at {goal_pos}")
        self.move_to_goal(goal_pos, goal_actions)
       
        # Check if the goal estimate is correct
        if self.robot.getLoc() == goal_pos:
            print(f"Reached {goal_pos}")
            if self.robot.getLoc() == self.game._goal:
                return True
            else:
                return False
        else:
            print(f"Didn't reach {goal_pos}, at {self.robot.getLoc()}")
            return False
    
    def run_solver(self, display=False):
        """implement the algorithm"""
        for i in range(5000):
            self.game.tick()
            mask = self.create_world_mask()
            world_estimate = self.world_estimate_net.runNetwork(self.game.exploredMap, mask)
            digit_estimate = self.digit_estimate_net.runNetwork(world_estimate)

            print(f"{self.game.getIteration()}: Robot at: {self.robot.getLoc()}, Score = {self.game.getScore()}")
        
            if self.ready_to_guess(digit_estimate):
                guess = self.make_guess(digit_estimate)
                print(f"Making guess after {self.game.getIteration()} iterations: {guess}")
                found_goal = self.check_goal(guess)
                if found_goal:
                    print(f"Found Gorrect Goal")
                    print(f"Final Score: {self.game.score}")
                    break
                else:
                    print(f"Found incorrect goal, continuing to search")
                    continue
        
        print(f"Final Guess: {digit_estimate.argmax()}")

        if display == True:
            self.show_images(world_estimate)
        return
            
    def show_images(self, world_estimate):
        Image.fromarray(world_estimate).show()
        Image.fromarray(np.uint8(self.game.exploredMap)).show()

if __name__ == "__main__":
    # The GreedyNavigator class makes the robot move in the direction that maximizes pixel values
    # map = Map()
    # rewards = []
    # runtimes = []
    # for i in range(10):
    #     start = time.time()
    #     print(f"Trial {i}")
    #     map.getNewMap()
    #     # solver = Solver(EntropyNavigator(), map)
    #     # solver = Solver(GreedyNavigator(), map)
    #     solver = Solver(LookaheadNavigator(), map)
    #     solver.run_solver(display=False)
    #     rewards.append(solver.game.score)
    #     end = time.time()
    #     runtime = end-start
    #     runtimes.append(runtime)
    # print(rewards)
    # print(f"Average: {np.mean(np.array(rewards))}")
    # print(runtimes)
    # print(f"Average: {np.mean(np.array(runtimes))}")

    scores = [20, 42, -804, -1553, 60, -119, -7677, -8719, -368]
    print(np.mean(scores))

   