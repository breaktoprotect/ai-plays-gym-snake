import gym
import gym_snake

import collections as coll
import time
import numpy as np 
import collections as coll
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import winsound
import keyboard


# Basic env set up
env = gym.make('snake-v0', render=True)
env.reset()
FREQUENCY = False # False (infinity), 1 to 100000000

# Training config
learning_rate = 1e-3 # or 0.001
goal_steps = 1000000

#* Generate random games
def generate_population(num_games=1000,score_requirement=0,save=True, watch=False, human=False, model_path=None):
    # Validation
    if human and model_path:
        print("[!] Error. Can't have both 'human' and 'model' turned on together.")
        return
    
    training_data = []
    scores = []
    accepted_scores = []

    if model_path:
        model = neural_network_model(10, 4)
        model.load(model_path)

    print("[*] Generate Initial Population started.")
    for eps in range(num_games):
        # Reset all parameters
        env.reset()
        game_memory = []
        score = 0
        prev_observation = []
        choices = []
        #facing = 0 #TODO: start with UP for now

        for _ in range(goal_steps):
            #? debug
            #input()
            if watch:
                env.render()
                time.sleep(1/8)
            
            #* Using a real human player to train the inital data
            if human:
                env.render()
                action = -1

                while action < 0:
                    time.sleep(0.1)

                    if keyboard.is_pressed('up'):
                        action = 0
                    elif keyboard.is_pressed('right'):
                        action = 1
                    elif keyboard.is_pressed('down'):
                        action = 2
                    elif keyboard.is_pressed('left'):
                        action = 3
                    
            else:
                action = env.action_space.sample() # 0 to 3; 0 up, 1 right, 2 down, 3 left

            #* Using previously trained model to play and generate training data
            if model_path and prev_observation:
                action = np.argmax(model.predict(np.array(prev_observation).reshape(-1, len(prev_observation), 1))[0])

            #facing = action
            observation, reward, done, info = env.step(action)

            #? debug
            if watch: print("observation, reward, done, info",observation, reward, done, info)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
                choices.append(action)

            prev_observation = observation
            score += reward

            # Terminate when game has ended
            if done:
                break
        
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 0: # data[1] is action; where data[0] is prev_observation
                    output = [1, 0, 0, 0] # Up
                elif data[1] == 1:
                    output = [0, 1, 0, 0] # Right
                elif data[1] == 2:
                    output = [0, 0, 1, 0] # Down
                elif data[1] == 3:
                    output = [0, 0, 0, 1] # Left

                # Training data instance is 1) Previous Observation; 2) Action carried out
                training_data.append([data[0], output])
        scores.append(score)

        # For progression tracing
        if (eps+1) % 500 == 0:
            print("[*]", eps+1, "has completed...")
    if save:
        training_data_save = np.array(training_data)
        if model_path:
            np.save('model_training_{GAMES}_{SCORE}.npy'.format(GAMES=num_games, SCORE=score_requirement), training_data_save)
        else:
            np.save('initial_training_{GAMES}_{SCORE}.npy'.format(GAMES=num_games, SCORE=score_requirement), training_data_save)

    print("Average overall score:", np.mean(scores))
    print("---------------------")
    print('Average accepted score:', np.mean(accepted_scores))
    print('Median accepted score:', np.median(accepted_scores))
    print(coll.Counter(accepted_scores))

    return training_data

#* Defining the neural network
def neural_network_model(input_size, output_size=4):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation="relu") # activation function = rectified linear
    network = dropout(network,0.8) # 80% retention

    network = fully_connected(network, 128, activation="relu")
    network = dropout(network,0.8) 

    network = fully_connected(network, output_size, activation='linear') # 4 outputs or actions of agent
    network = regression(network, optimizer='adam', learning_rate=learning_rate, loss='mean_square', name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

#* Train the model
def train_model(training_data, num_of_epoch=1, model=False): 
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1) # Observations
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]), output_size=4)

    model.fit({'input': X}, {'targets': y}, n_epoch=num_of_epoch,snapshot_step=1000, show_metric=True, run_id='openaistuff')

    return model

#* Play using trained model
def model_play_game(model_path, num_of_games, frequency=5,watch=False):
    env = gym.make('snake-v0', render=True)
    model = neural_network_model(10, 4)
    model.load(model_path)
    scores = []
    choices = []

    for each_game in range(num_of_games):
        score = 0
        game_memory = []
        prev_observation = []
        env.reset()

        for _ in range(goal_steps):
            if frequency:
                time.sleep(1/frequency)

            if watch:
                env.render()

            if len(prev_observation) == 0:
                action = env.action_space.sample()
            else:
                #debug - try
                action = np.argmax(model.predict(np.array(prev_observation).reshape(-1, len(prev_observation), 1))[0])
            choices.append(action)


            new_observation, reward, done, info = env.step(action)

            #debug
            if watch:
                print("new_observation, reward, done, info-->", new_observation, reward, done, info )

            prev_observation = new_observation
            game_memory.append([prev_observation, action])
            score += reward
            
            if done:
                if watch:
                    time.sleep(1)
                break
                #time.sleep(1)
                #input()

        scores.append(score)

    print("Average overall score:", np.mean(scores))
    print("---------------------")
    #print('Average accepted score:', np.mean(accepted_scores))
    #print('Median accepted score:', np.median(accepted_scores))
    print('Choices:',coll.Counter(choices))

    return


###* MAIN PROGRAM ###
def main():
    # True or False to turn on or off respectively
    IS_HUMAN_TRAINING = False
    IS_INITIAL_GENERATE= False
    IS_TRAIN = False
    IS_PLAY = True
    IS_FURTHER_GENERATE = False
    
    #! Train with human
    if IS_HUMAN_TRAINING:
        training_data = generate_population(num_games=3, score_requirement=0, human=True)

    #* Generate initial games
    if IS_INITIAL_GENERATE:
        training_data = generate_population(num_games=1000, score_requirement=1)
        winsound.Beep(1500,500)
    
    #* Training historical
    if IS_TRAIN:
        training_data = np.load("model_training_1000_20.npy", allow_pickle=True)

        model = train_model(training_data, num_of_epoch=10)
        model.save('gen2_1000_20_epoch_10.model')
        winsound.Beep(1500,250);time.sleep(0.1);winsound.Beep(1500,500)
    

    #* Test the game
    if IS_PLAY:
        model_play_game(model_path='gen2_1000_20_epoch_10.model', num_of_games=20, watch=True, frequency=1000000) #frequency=5)

    #* Model to generate more games and training data
    if IS_FURTHER_GENERATE:
        training_data = generate_population(num_games=1000, score_requirement=20, model_path="3_epoch_5.model")
        winsound.Beep(1500,500)
    

if __name__ == "__main__":
    main()