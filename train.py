"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=2000000)
    parser.add_argument("--replay_memory_size", type=int, default=50000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):

    #if cuda availabe seed it so that it gives consistent operations for reprodiction
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    model = DeepQNetwork()
    
    #In summary, this code snippet is used to ensure that the log directory is clean and ready for writing new logs using SummaryWriter.
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)



    #define optimize and loss function for the model
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    #get the first game to and the corresponding frame and detsails 
    #process that image to the form the model takes input in
    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image)#convert form numpy to torch tensor since both opencv and PIL handle images in numpy



    if torch.cuda.is_available():
        #we must load both not only one
        model.cuda()
        image = image.cuda()


    #here it is replicating four times becaue model is defined to take in input of size 4
    #additional dimention is added to make it a batch since pytorch need inputs in batches
        #here each each batch has only one image but in other cases they could be more
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    replay_memory = []
    iter = 0
    while iter < opt.num_iters:#its acutaul iters i.e here time frames
        prediction = model(state)[0]


        # Exploration or exploitation
        #By using this epsilon-greedy strategy, the agent balances between exploring new actions (to discover potentially better strategies) and exploiting known good actions (to maximize reward).

        #Epsilon is a hyperparameter that determines the probability of choosing a random action (exploration) versus selecting the action with the highest estimated value (exploitation). The equation starts with opt.final_epsilon, which is the minimum value of epsilon (typically close to 0), and linearly decays towards opt.initial_epsilon over opt.num_iters iterations. This means that in the beginning, the agent is more exploratory (higher epsilon), but as training progresses, it becomes more exploitative (lower epsilon).
        epsilon = opt.final_epsilon + (
                (opt.num_iters - iter) * (opt.initial_epsilon - opt.final_epsilon) / opt.num_iters)
        u = random()#decide whether to take a random action or not.
        random_action = u <= epsilon
        if random_action:
            print("Perform a random action")
            action = randint(0, 1)#here if its lees than 1 game will handle it as zero
        else:
            action = torch.argmax(prediction).item()#her outputs can be only 1 or 0


        next_image, reward, terminal = game_state.next_frame(action)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size,
                                    opt.image_size)
        next_image = torch.from_numpy(next_image)


        if torch.cuda.is_available():
            next_image = next_image.cuda()

        #store the state,action,next_state : reward 
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]#this line tranforms the dimension since next_image dim is width x height but we know state shouldof dimension 4 because of model
        replay_memory.append([state, action, reward, next_state, terminal])
        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[0]
        
        #sample a batch from the replay memory
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        
        
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(
            np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        current_prediction_batch = model(state_batch)
        next_prediction_batch = model(next_state_batch)

        #target Q-value calulated for the batch using bellman equation .sum of batch is taken
        y_batch = torch.cat(
            tuple(reward if terminal else reward + opt.gamma * torch.max(prediction) for reward, terminal, prediction in
                  zip(reward_batch, terminal_batch, next_prediction_batch)))
        #q-value predicted by the model for curren batch. sum of the batch is taken
        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
        
        
        optimizer.zero_grad()#this can be done before or after gradient calculation but it jus clears all gradients from previous steps else loss.backward will all to prevsious gradients
        # y_batch = y_batch.detach()
        loss = criterion(q_value, y_batch) #using crsoos entropy loss is calculated
        loss.backward() #gradients of loss wrt each parameter is calculated
        optimizer.step() #and optimum step is taken

        #update the state to next state
        state = next_state
        iter += 1

        #print the values
        print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
            iter + 1,
            opt.num_iters,
            action,
            loss,
            epsilon, reward, torch.max(prediction)))
        writer.add_scalar('Train/Loss', loss, iter)
        writer.add_scalar('Train/Epsilon', epsilon, iter)
        writer.add_scalar('Train/Reward', reward, iter)
        writer.add_scalar('Train/Q-value', torch.max(prediction), iter)

        #save model every 1000000 iteration
        if (iter+1) % 1000000 == 0:
            torch.save(model, "{}/flappy_bird_{}".format(opt.saved_path, iter+1))


    torch.save(model, "{}/flappy_bird".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
