################################################################
## This file contains the main code to train an agent to play ##
## the Snake game using deep learning with neural networks.   ##
################################################################

import torch
import random
import os
import threading
import numpy as np
from collections import deque
from game import SnakeGameAI,Direction,Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
MODEL_PATH = "model/model.pth"

class Agent:

    def __init__(self):
        #init the agent
        self.n_games = 0
        self.epsilon = 0 
        self.gamma = 0.9 
        self.memory = deque(maxlen=MAX_MEMORY) 
        self.model = Linear_QNet(11,256,3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        if torch.cuda.is_available():
            self.model.cuda()
        if os.path.isfile(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH))

    def get_state(self, game):
        # Get the current state of the game
        head = game.snake[0]
        point_l = Point(head.x-20, head.y)
        point_r = Point(head.x+20, head.y)
        point_u = Point(head.x, head.y-20)
        point_d = Point(head.x, head.y+20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            #danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            #danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            #move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #food location
            game.food.x < game.head.x, #food left
            game.food.x > game.head.x, #food right
            game.food.y < game.head.y, #food up
            game.food.y > game.head.y #food down
        ]
        return np.array(state,dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # Store the past experience in memory
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        # Train the neural network using past experience
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE) 
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Train the neural network using recent experience
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Gets the action to take, with epsilon-greedy exploration
        self.epsilon = 80-self.n_games
        final_move = [0,0,0]
        if random.randint(0,200)<self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 =  torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0) 
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    plot_score = []
    plot_mean_score = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state 
        state_old = agent.get_state(game)

        #get move
        final_move = agent.get_action(state_old)

        #perform move
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #remeber
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            #train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print("Game", agent.n_games, "Score", score, "Record", record)

            plot_score.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_score.append(mean_score)
            plot(plot_score, plot_mean_score)

if __name__ == "__main__":
    train()