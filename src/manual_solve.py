#!/usr/bin/python

# ASSIGNMENT -3 (1MAI)
# GROUP SUBMISSION
# STUDENT - HUZEFA MANSOOR LOKHANDWALA (21241263)
# STUDENT - HARSHITHA BENGALURU RAGHURAM (21235396)
# GITHUB REPO - https://github.com/huzz/ARC

import os, sys
import json
import numpy as np
import re

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.


def solve_c8cbb738(x):
    # LEVEL : DIFFICULT
    #file: solve_c8cbb738.json
    
    #Get the the unique colour numbers and the counts of each colour
    (colour_list, colour_counts)=np.unique(x, return_counts=True)
    #Find the colour that has maximum count, which is our background colour
    background_colour = colour_list[np.argmax(colour_counts)]
    #Get the dimension of our output array. 
    #Iterating through the unique colours except the background colour and returning the dimension wof maximum colour
    dimension = max([(max(np.where(x==colour)[0])-min(np.where(x==colour)[0])) for colour in colour_list if colour!=background_colour]) + 1
    #cretaing an np array of zeros with output dimension
    output_array = np.zeros((dimension,dimension))
    #for each unique colour
    for colour in colour_list:
        #except the background colour
        if colour!=background_colour:
            #getting the colour row and column coordinates
            colour_row,colour_column  = np.where(x==colour)
            #slicing the array with the min and max coordinates of the colour
            #here we get the sliced array according to the colour coordinates
            colour_array = x[min(colour_row):(max(colour_row) + 1),min(colour_column):(max(colour_column) + 1)]
            #making every pixel in the colour_array '0' if its not of the main colour
            colour_array=np.where(colour_array!=colour, 0, colour_array)      
            #if the rows of the colour are less than rows of the output array
            if colour_array.shape[0]<output_array.shape[0]:
                #finding the pad width inorder to add inorder to make the dimension same as the ouput array
                pad_width = (output_array.shape[0]-colour_array.shape[0])-(output_array.shape[0]-colour_array.shape[0])//2
                #adding a pad on the top and bottom of the array
                colour_array = np.pad(colour_array, ((pad_width, pad_width), (0, 0)), mode='constant')
            #if the columns of the colour are less than columns of the output array
            elif colour_array.shape[1]<output_array.shape[1]:
                #finding the pad width inorder to add inorder to make the dimension same as the ouput array
                pad_width = (output_array.shape[1]-colour_array.shape[1])-(output_array.shape[1]-colour_array.shape[1])//2
                #adding a pad on the left and right of the array
                colour_array = np.pad(colour_array, ((0, 0), (pad_width, pad_width)), mode='constant')
            #simply adding the colour array to the output array 
            output_array+=colour_array
                
    #replacing all "0" with the background colour
    output_array=np.where(output_array==0, background_colour, output_array)
    
    #returning the final output value
    return output_array

def solve_1b60fb0c(x):
    # LEVEL : DIFFICULT
    #file: solve_1b60fb0c.json
    
    #storing the original dimensions of x
    ori_dim=x.shape
    #storing the first row, last row and last column
    first_row = x[0]
    last_row = x[-1]
    last_column = x[:,-1]
    
    #initializing the pad_type as None
    #pad type can have 4 values ["upper_right", "lower_right", "upper", "lower"]
    pad_type=None
    # if the top row and bottom row are not equal and the last column is not full of 0 values
    if len(set(first_row))!=len(set(last_row)) and list(set(last_column))!=[0]:
        # if the first row is not full of 0 values and setting the respective flagz
        if list(set(first_row))!=[0]:
            #adding a pad to the top row and last column and setting the respective flag
            x=np.pad(x,((1,0),(0,1)))
            pad_type="upper_right"
        # if the last row is not full of 0 values
        else:
            #adding a pad to the bottom row and last column and setting the respective flag
            x=np.pad(x,((0,1),(0,1)))
            pad_type="lower_right"
    # if the top row and bottom row are not equal and the last column is full of 0 values
    elif len(set(first_row))!=len(set(last_row)) and list(set(last_column))==[0]:
        # if the first row is not full of 0 values
        if list(set(first_row))!=[0]:
            #adding a pad to the top row and setting the respective flag
            x=np.pad(x,((1,0),(0,0)))
            pad_type="upper"
        # if the last row is not full of 0 values
        else:
            #adding a pad to the bottom row and setting the respective flag
            x=np.pad(x,((0,1),(0,0)))
            pad_type="lower"
        
    #getting Upper triangle of an array and the lower part values are set to "0"
    diag_up = np.triu(x)
    #rotating the array 2 times by 0 degrees
    diag_up_T = np.rot90(diag_up, 2)
    #adding the diagonal upper triangle array and the rotated version
    output_array = diag_up+diag_up_T
    #replacing the values having value 2 with 1. 
    #This will make the entire shape except the red values
    output_array=np.where(output_array==2, 1, output_array)         
    #removing the difference with the original array to get the positions of red values
    diff_array=output_array-x
    #adding the difference array with the output array
    output_array+=diff_array
    
    #Removing the additional pads depending on the flag set
    if pad_type=='upper_right':output_array=output_array[output_array.shape[0]-ori_dim[0]:,:-(output_array.shape[1]-ori_dim[1])]
    elif pad_type=='lower_right':output_array=output_array[:-(output_array.shape[0]-ori_dim[0]),:-(output_array.shape[1]-ori_dim[1])]
    elif pad_type=='upper':output_array=output_array[(output_array.shape[0]-ori_dim[0]):,:]
    elif pad_type=='lower':output_array=output_array[:-(output_array.shape[0]-ori_dim[0]),:]
        
    return output_array

            

def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__": main()

