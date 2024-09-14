from flying import FlyingCycle
from plotting import updateplots
from collections import deque
import torch, numpy as np, random, enum, math
from model import Linear_QNet, QTrainer

MAX_MEMORY = 10000
BATCH_SIZE = 100
LEARNING_RATE = 0.00001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(7, 32, 3)
        self.trainer = QTrainer(self.model, LEARNING_RATE, self.gamma)
        # model, trainer

    def get_state(self, game):
        return game.getState()

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def train_long(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, new_state, done in sample:
        #    self.trainer.train_step(state, action, reward, new_state, done)

    def train_short(self, state, action, reward, new_state, done):
        self.trainer.train_step(state, action, reward, new_state, done)

    def get_input(self, state):
        self.epsilon = 10000 - self.n_games
        final_input = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            input = random.randint(0, 2)
            final_input[input] = 1
        elif random.random() < 0.02:
            input = random.randint(0, 2)
            final_input[input] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            input = torch.argmax(prediction).item()
            final_input[input] = 1

        return final_input

def train():
    plot_scores = []
    plot_mean = []

    currentFlight = []
    bestFlight = []

    total = 0
    record = 0

    agent = Agent()
    game = FlyingCycle()

    while True:
        old_state = agent.get_state(game)
        input = agent.get_input(old_state)
        reward, done, score, inputlist = game.nextFrame(input)
        new_state = agent.get_state(game)

        currentFlight.append((old_state[1], 430 - old_state[2]))

        agent.train_short(old_state, input, reward, new_state, done)
        agent.remember(old_state, input, reward, new_state, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long()

            if score > record:
                record = score
                bestFlight = currentFlight
                print("New best!", old_state[1], "in", old_state[0], "frames.", str(round(old_state[1]/(old_state[0]+40), 2)))
                filename = str(record)
                file = open('E:\\' + filename + ".txt",'w')
                file.write(str(inputlist))
                file.close()
                # agent.model.save()

            print("Game", agent.n_games, "Score", score, "Record", record)

            plot_scores.append(score)
            total += score
            mean = total / agent.n_games
            plot_mean.append(mean)

            updateplots(plot_scores, plot_mean, currentFlight, bestFlight)
            currentFlight = []

train()