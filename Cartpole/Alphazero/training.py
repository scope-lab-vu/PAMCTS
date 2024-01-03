"""
Alphazero Cartpole Training
"""

"""
import modules
"""
import configparser
import multiprocessing
import gym
import time
import os
import random
import numpy as np
import pandas as pd
from MCTS.MCTS import MCTS


def build_model(num_hidden_layers):
    """
    build the model for specific number of hidden layers
    """
    import tensorflow as tf
    from keras.layers import Input, Dense
    from keras.models import Model
    from tensorflow.keras.optimizers import Adam
    # tensors
    global layer
    state = Input(shape=(4,))

    # create num_hidden_layers layers
    for i in range(int(num_hidden_layers)):
        if i == 0:
            layer = Dense(units=4, activation=tf.nn.relu, name='Layer'+str(i))(state)
        else:
            layer = Dense(units=4, activation=tf.nn.relu, name='Layer'+str(i))(layer)

    # outputs
    pi_hat = Dense(units=4, activation=tf.nn.softmax, name='pi_hat')(layer)
    v_hat = Dense(units=1, name='v_hat')(layer)

    # construct the model
    model = Model(inputs=state, outputs=[pi_hat, v_hat])
    model.compile(loss={'pi_hat': 'kl_divergence', 'v_hat': 'mse'}, optimizer=Adam())
    return model


def training_data_generation_cartpole(q, shared_result, args_dict):
    """
    training data generation process
    """
    import keras
    model_file = args_dict['model_file']
    num_iterations = args_dict['num_iterations']
    c_puct = args_dict['c_puct']
    gamma = args_dict['gamma']

    env = gym.make('CartPole-v1')
    env._max_episode_steps = 2500
    start_time = time.time()
    print("ID of process running data generation: {}".format(os.getpid()))
    text_file = '../../data/training_data_cartpole.csv'

    while 1:
        flag = False
        network = None
        # load the model
        lock = multiprocessing.Lock()
        lock.acquire()
        if os.path.isdir(model_file):
            network = keras.models.load_model(model_file)
        else:
            flag = True
        lock.release()

        if flag:
            time.sleep(1)
            continue

        curr_state = env.reset()
        search_agent = MCTS(current_state=curr_state, gamma=gamma, num_iter=num_iterations, c_puct=c_puct)
        terminated = False
        cum_reward = 0

        while not terminated:
            action = search_agent.run_mcts(network=network, env=env)

            # get the training data
            training_data = search_agent.get_training_data()
            q.put(training_data)

            curr_state, reward, terminated, _ = env.step(action)
            search_agent.mcts_update_root(curr_state)
            print(os.getpid(), cum_reward, curr_state, reward, terminated, action)
            cum_reward += 1

            if cum_reward > 2500:
                terminated = True
            if terminated:
                shared_result.append([time.time() - start_time, cum_reward])
                df = pd.DataFrame(np.array(shared_result), columns=['time', 'cum_reward'])
                if os.path.isfile(text_file):
                    df.to_csv(text_file, mode='a', header=False, index=False)
                else:
                    df.to_csv(text_file, mode='a', header=True, index=False)


def nn_training(q, args_dict):
    """
    train the neural network
    """
    import keras
    start_time = time.time()
    print("ID of process running DNN training: {}".format(os.getpid()))
    replay_buffer = []
    training_text = 'training_frozenlake_history.txt'
    model_file = args_dict['model_file']
    num_hidden_layers = args_dict['num_hidden_layers']

    network = build_model(num_hidden_layers)
    network.save(model_file)

    while 1:
        lock = multiprocessing.Lock()
        lock.acquire()
        network = keras.models.load_model(model_file)
        lock.release()

        while len(replay_buffer) < 200:
            training_data = q.get()
            replay_buffer.append(training_data)

        # for one call fit, update replay buffer for 100 training data
        # outer for loop 10, inner loop 5
        for m in range(10):
            for i in range(100):
                # maybe later change to a larger size of buffer
                if len(replay_buffer) > 10000:
                    training_data = q.get()
                    replay_buffer.pop(0)
                    replay_buffer.append(training_data)
                else:
                    training_data = q.get()
                    replay_buffer.append(training_data)

            for n in range(5):
                # define input and output batches
                states = []
                pi_hats = []
                v_hats = []

                # use sample 200 as minibatch among 10000
                minibatch = random.sample(replay_buffer, 200)

                for training_data in minibatch:
                    states.append(training_data[0])
                    pi_hats.append(training_data[1])
                    v_hats.append(training_data[2])

                states = np.array(states).reshape((200, 4))
                pi_hats = np.array(pi_hats).reshape((200, 2))
                v_hats = np.array(v_hats).reshape((200, 1))

                # call fit for 5 times and then save the model
                # change epochs=1 and 5 compare these two
                history = network.fit(x=states, y=[pi_hats, v_hats], batch_size=50, epochs=1)
                with open(training_text, 'a') as f:
                    line = str(time.time() - start_time) + ' : ' + str(history.history['loss'])
                    f.write(f"{line}\n")

        lock = multiprocessing.Lock()
        lock.acquire()
        network.save(model_file)
        lock.release()


if __name__ == '__main__':
    # read from config.yml file
    config = configparser.ConfigParser()
    config.read('../../config.yml')
    num_hidden_layers = int(config.get('Alphazero_Cartpole_Training', 'num_hidden_layers'))
    num_data_generation_processes = int(config.get('General', 'num_cpus')) - 2
    num_iterations = int(config.get('Alphazero_Cartpole_Training', 'num_iterations'))
    c_puct = float(config.get('Alphazero_Cartpole_Training', 'c_puct'))
    gamma = float(config.get('Alphazero_Cartpole_Training', 'gamma'))

    args_dict = {'model_file': "../../data/alphazero_cartpole", 'num_hidden_layers': num_hidden_layers,
                'num_data_generation_processes': num_data_generation_processes, 'num_iterations': num_iterations,
                'c_puct': c_puct, 'gamma': gamma}

    q = multiprocessing.Queue()
    data_generation_processes = []

    with multiprocessing.Manager() as manager:
        shared_result = manager.list([])
        for i in range(num_data_generation_processes):
            process = multiprocessing.Process(target=training_data_generation_cartpole,
                                              args=(q, shared_result, args_dict))
            process.start()
            data_generation_processes.append(process)

        process_nn_training = multiprocessing.Process(target=nn_training, args=(q, args_dict))
        process_nn_training.start()

        # should train infinitely until converges to maximum cumulative reward
        for process in data_generation_processes:
            process.join()
        process_nn_training.terminate()
