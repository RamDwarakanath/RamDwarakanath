#goals
# grid with randomly generated maze for training and testing
# object needs to get to target
# @training_function on line 238
# Issues
# - Currently, cannot go back on path so if it gets trapped by 3 surrounding walls it fails (If it goes back on path it gets stuck in infinite loop)
#   - FIX: Allow position to repeat but not position pair, so no infinite loop is created but it is still allowed to go back and find a better solution.
# - Performance not good when trained on randomly generated maze
#   - FIX: Created smaller maze that includes every possible state scenario to increase performance
# Performance:
# {Trained on 7x7 grid with 2 random walls, Accuracy: 94% on 40x40 grid with 150 random walls, lr=0.001, gamma=0.9, epsilon=0.5}
# {Trained on 7x7 grid with pre-determined walls, Accuracy: 96.8% on 40x40 grid with 150 random walls, lr=0.001, gamma=0.9, epsilon=0.8} 

# Import libraries
import numpy as np
from matplotlib import pyplot

# Environment Size
# environment_rows = 19 #train
# environment_columns = 19 #train
environment_rows = 40 #test
environment_columns = 40 #test

# Environment Rewards
rewards = np.full([environment_rows, environment_columns], -1)
for i in range(len(rewards)):
    rewards[i][0] = -100
    rewards[i][-1] = -100
for i in range(len(rewards[0])):
    rewards[0][i] = -100
    rewards[-1][i] = -100

# define state
# (food_dir_x, food_dir_y, wall_x, wall_y)
# food_dir is +1 when coordinate decreases i.e. left or up
# wall is +1 when it is to the left or above

# Q values
NUM_FOOD_DIR_X = 3
NUM_FOOD_DIR_Y = 3
NUM_WALL_DIR_X = 4
NUM_WALL_DIR_Y = 4
NUM_ACTIONS = 4
q_values = np.zeros((NUM_FOOD_DIR_X, NUM_FOOD_DIR_Y, NUM_WALL_DIR_X, NUM_WALL_DIR_Y, NUM_ACTIONS))

# Helper functions
def is_valid_state(row, column, reward_grid):
    if reward_grid[row, column] == -100 or reward_grid[row, column] == 100:
        return False
    else:
        return True
def next_action(position, state, rewards_wall, explore, path, q_values):
    # actions = ['up', 'right', 'down', 'left'] = [0, 1, 2, 3]]
    circles = False
    def next_position(position, action):
        if action == 0:
            new_position = (position[0]-1, position[1])
        elif action == 1:
            new_position = (position[0], position[1]+1)
        elif action == 2:
            new_position = (position[0]+1, position[1])
        elif action == 3:
            new_position = (position[0], position[1]-1)
        return new_position
    
    if explore:
        if np.random.random(1) > 0.8:
            action = np.random.randint(4)
        else:
            action = np.argmax(q_values[state])
    else:
        action = np.argmax(q_values[state])
        new_position = next_position(position, action)
        act_list = np.argsort(q_values[state])
        # i = -2
        if path.count(new_position) > 0:
            for i in [-2, -3, -4]:
                # check if pair exists
                for j in range(len(path)-1):
                    # print('path[-1] == path[j] and new_position == path[j+1]', path[-1], path[j], new_position, path[j+1])
                    if path[-1] == path[j] and new_position == path[j+1]: # if the pair exists in path
                        # print('i', i)
                        action = act_list[i]
                        new_position = next_position(position, action)
            # if i == -5:
            #     circles = True
            #     break
            # action = act_list[i]
            # # old_position = new_position
            # new_position = next_position(position, action)
            # i -= 1
        
        return new_position, action, circles

    new_position = next_position(position, action) 
    return new_position, action
def state(position, food, rewards):
    # (food_dir_x, food_dir_y, wall_x, wall_y)
    # food_dir is +1 when coordinate decreases i.e. left or up
    # wall is +1 when it is to the left or above
    # 0 if no walls, 1 if wall left or up, 2 if wall right or down, 3 if wall both

    food_dir_x = 0
    food_dir_y = 0
    wall_x = 0
    wall_y = 0

    if position[1] - food[1] == 0:
        food_dir_x = 0
    elif position[1] - food[1] >= 1:
        food_dir_x = -1
    elif position[1] - food[1] <= -1:
        food_dir_x = 1
    
    if position[0] - food[0] == 0:
        food_dir_y = 0
    elif position[0] - food[0] >= 1:
        food_dir_y = 1
    elif position[0] - food[0] <= -1:
        food_dir_y = -1


    # if each x and y direction is out of bounds only look at the other direction. 

    # (food_dir_x, food_dir_y, wall_x, wall_y)
    # food_dir is +1 when coordinate decreases i.e. left or up
    # wall is +1 when it is to the left or above
    # 0 if no walls, 1 if wall left or up, 2 if wall right or down, 3 if wall both
    if position[1] < environment_columns-1 and position[0] < environment_rows-1:
        if rewards[position[0]-1, position[1]] == -100 and rewards[position[0]+1, position[1]] == -100:
            wall_y = 3 
        elif rewards[position[0]-1, position[1]] == -100:
            wall_y = 1
        elif rewards[position[0]+1, position[1]] == -100:
            wall_y = 2
        if rewards[position[0], position[1]-1] == -100 and rewards[position[0], position[1]+1] == -100:
            wall_x = 3
        elif rewards[position[0], position[1]-1] == -100:
            wall_x = 1
        elif rewards[position[0], position[1]+1] == -100:
            wall_x = 2
    elif position[0] == 0 or position[0] == environment_rows:
        wall_x = 3
        wall_y = 1
    elif position[1] == 0 or position[1] == environment_columns:
        wall_x = 1
        wall_y = 3 

    return (food_dir_x, food_dir_y, wall_x, wall_y)

# Hyperparameters
NUM_EPOCHS = 50000
learning_rate = 0.001
discount_factor = 0.9

def training():
    # Training
    for epoch in range(NUM_EPOCHS):
        
        # create random walls in grid
        rewards_wall = rewards.copy()
        
        # for i in range(10):
        #     a = (np.random.randint(1, environment_rows-1), np.random.randint(1, environment_columns-1))
        #     while not is_valid_state(a[0], a[1], rewards_wall):
        #         a = (np.random.randint(1, environment_rows-1), np.random.randint(1, environment_columns-1))
        #     rewards_wall[a[0]][a[1]] = -100

        rewards_wall[9][6] = -100        
        rewards_wall[9][12] = -100
        rewards_wall[6][9] = -100        
        rewards_wall[12][9] = -100

        rewards_wall[5][8] = -100        
        rewards_wall[5][10] = -100 
        rewards_wall[13][8] = -100        
        rewards_wall[13][10] = -100

        rewards_wall[8][5] = -100        
        rewards_wall[8][13] = -100 
        rewards_wall[10][5] = -100        
        rewards_wall[10][13] = -100  

        rewards_wall[9][9] = -100        
        rewards_wall[10][9] = -100 
        rewards_wall[8][9] = -100        
        rewards_wall[9][10] = -100   
        rewards_wall[9][8] = -100    

        # create random start point that is valid
        position = (np.random.randint(1, environment_rows-1), np.random.randint(1, environment_columns-1)) 
        while not is_valid_state(position[0], position[1], rewards_wall):
            position = (np.random.randint(1, environment_rows-1), np.random.randint(1, environment_columns-1))
        
        # create random food location that is valid
        food = (np.random.randint(1, environment_rows-1), np.random.randint(1, environment_columns-1))
        while not is_valid_state(food[0], food[1], rewards_wall):
            food = (np.random.randint(1, environment_rows-1), np.random.randint(1, environment_columns-1))
        rewards_wall[food[0]][food[1]] = 100

        while is_valid_state(position[0], position[1], rewards_wall):
            
            # take next step based on policy
            old_position = position
            old_state = state(old_position, food, rewards_wall)
            position, action = next_action(position, old_state, rewards_wall, True, [], q_values)

            # print('old_position', old_position)
            # print('old_state', old_state)
            # print('new position', position)
            # print('action', action)

            # update q value
            # print('pos', position)
            new_state = state(position, food, rewards_wall)
            # print(old_state, action)
            old_state_action = old_state[0], old_state[1], old_state[2], old_state[3], action
            old_q_value = q_values[old_state_action]

            temporal_difference = rewards_wall[position] + discount_factor * np.max(q_values[new_state]) - old_q_value
            q_values[old_state_action] = old_q_value + learning_rate * temporal_difference

def is_training(train):
    if train:
        training()
        np.save('/Users/rdwarakanath/Desktop/Fun/AI/Reinforcement_Learning/Maze/q_values', q_values)

def get_shortest_path():
    q_values = np.load('/Users/rdwarakanath/Desktop/Fun/AI/Reinforcement_Learning/Maze/q_values.npy')
    rewards_wall = rewards.copy()

    # Random walls generated (comment out if using test grid below)
    for i in range(150):
        a = (np.random.randint(1, environment_rows-1), np.random.randint(1, environment_columns-1))
        while not is_valid_state(a[0], a[1], rewards_wall):
            a = (np.random.randint(1, environment_rows-1), np.random.randint(1, environment_columns-1))
        rewards_wall[a[0]][a[1]] = -100

    # # Showing test grid 
    # rewards_wall[9][6] = -100        
    # rewards_wall[9][12] = -100
    # rewards_wall[6][9] = -100        
    # rewards_wall[12][9] = -100

    # rewards_wall[5][8] = -100        
    # rewards_wall[5][10] = -100 
    # rewards_wall[13][8] = -100        
    # rewards_wall[13][10] = -100

    # rewards_wall[8][5] = -100        
    # rewards_wall[8][13] = -100 
    # rewards_wall[10][5] = -100        
    # rewards_wall[10][13] = -100  

    # rewards_wall[9][9] = -100        
    # rewards_wall[10][9] = -100 
    # rewards_wall[8][9] = -100        
    # rewards_wall[9][10] = -100   
    # rewards_wall[9][8] = -100   

    path = []

    #food loc
    food = (np.random.randint(1, environment_rows-1), np.random.randint(1, environment_columns-1))
    while not is_valid_state(food[0], food[1], rewards_wall):
        food = (np.random.randint(1, environment_rows-1), np.random.randint(1, environment_columns-1))
    rewards_wall[food[0]][food[1]] = 100

    #start state
    position = (np.random.randint(1, environment_rows-1), np.random.randint(1, environment_columns-1)) 
    while not is_valid_state(position[0], position[1], rewards_wall):
        position = (np.random.randint(1, environment_rows-1), np.random.randint(1, environment_columns-1))
    start_position = position

    path.append(position)
    while is_valid_state(position[0], position[1], rewards_wall):
        state_cur = state(position, food, rewards_wall)
        # print(rewards_wall)
        # print(food)
        # print(path)
        position, action, circles = next_action(position, state_cur, rewards_wall, False, path, q_values)
        if circles == True:
            break
        path.append(position)

    for coord in path[:-1]:
        rewards_wall[coord] = 50
    # for coord in walls:
    #     rewards_wall[coord] = -100

    return [start_position, food, path], rewards_wall




# Running training
is_training(False)

# Display Path in window 
def display(b):
    pyplot.figure(figsize=(7,7))
    pyplot.imshow(b)
    pyplot.show()

# # Running one test
# a, b = get_shortest_path()
# print(b)
# print(a)
# display(b)

# Run 1000 random tests to see accuracy
def check_accuracy():
    num_correct = 0
    total_tests = 0
    for i in range(1000):
        a, b = get_shortest_path()
        food = a[1]
        path = a[2]
        if food == path[-1]:
            num_correct += 1
        total_tests += 1
    print(f'Got {num_correct} / {total_tests} with accuracy {float(num_correct)/float(total_tests)*100}')

check_accuracy()