import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3
env = gym.make('CartPole-v0')
env.reset()
goal_steps =500

score_requirement = 50
initialGames=  8000 #cant make it too big or you will brute force every possibility

def someRandomGamesFirst():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render() #render to go faster
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if done:
                break

#someRandomGamesFirst()

def initialPopulation():
    trainingData = []
    scores = []
    acceptedScores =[]

    for _ in range(initialGames):
        score = 0
        gameMemory =[]
        prevObservation = []

        for _ in range(goal_steps):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)

            if len(prevObservation) > 0 :
                gameMemory.append([prevObservation,action])
            prevObservation = observation
            score += reward
            if done:
                break
        if score >= score_requirement:
            acceptedScores.append(score)
            for data in gameMemory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]

                trainingData.append([data[0],output])

        env.reset()
        scores.append(score)
    trainingDataSaved = np.array(trainingData)
    np.save("saved.npy", trainingDataSaved)

    print('average accepted score: ', mean(acceptedScores))
    print( 'median accepted scores: ', median(acceptedScores))
    print( Counter(acceptedScores))

    return trainingData


initialPopulation()

def neuralNetworkModel(inputSize):
    #layer 1
    network = input_data(shape = [None, inputSize, 1], name = 'input')
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network,0.8)
    #layer 2
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    #layer 3
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)
    #layer 4
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    #layer 5
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    # the two signifies the amount of actions provided
    network = fully_connected(network,2, activation = 'softmax' )
    network = regression(network,optimizer = 'adam', learning_rate = LR, loss='categorical_crossentropy', name = 'targets')

    model = tflearn.DNN(network, tensorboard_dir = 'log')

    return model

def train_model (trainingData, model=False):

    X = np.array([i[0] for i in trainingData]).reshape(-1,len(trainingData[0][0]),1)
    y = [i[1] for i in trainingData]

    if not model:
        model = neuralNetworkModel(inputSize=len(X[0]))

    model.fit({'input': X}, {"targets":y}, n_epoch = 3, snapshot_step = 500, show_metric = True, run_id ='openaicatPole')

    return model
trainingData = initialPopulation()
model = train_model(trainingData)


scores = []
choices = []

for each_game in range(10):
    score = 0
    game_memory=[]
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()

        if len(prev_obs)==0:
            action = random.randrange(0,2)

        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
        choices.append(action)

        new_observation, reward, done, info = env.step(action)

        prev_obs = new_observation
        game_memory.append([new_observation,action])
        score += reward
        if done:
            break

    scores.append(score)

print("average score: ", sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
print(" Score Requirement: " , score_requirement)


