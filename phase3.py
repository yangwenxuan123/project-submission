import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

training_set = pd.read_pickle('actual_attackers.pkl')
predicted_positions = pd.read_pickle('predicted_positions.pkl')

sq='sequence_'
for i in range(1,len(training_set)):#STATS DATASET
    sq=sq+str(i)
    #offense players values
    for row in range(0,len(training_set[sq])):
        ox = []
        oy = []
        dx = []
        dy = []
        bx = []
        by = []
        fig = plt.figure()
    
        for col in range(0,22):
            if col%2==0:
                print(training_set[sq][row][col])
                ox = np.append(ox,training_set[sq][row][col])
            else:
                print(training_set[sq][row][col])
                oy=np.append(oy,training_set[sq][row][col])
        plt.scatter(ox, oy, label= "Offense", color= "blue",	marker= "*", s=50) 
    
    #defense players values 
        for col in range(0,22):
            if col%2==0:
                print(predicted_positions[sq][row][col])
                dx = np.append(dx,predicted_positions[sq][row][col])
            else:
                print(predicted_positions[sq][row][col])
                dy=np.append(dy,predicted_positions[sq][row][col])
        plt.scatter(dx, dy, label= "Defence", color= "red",	marker= "*", s=50)
        
    #ball values
        for col in range(22,24):
            if col%2==0:
                print(training_set[sq][row][col])
                bx = np.append(bx,training_set[sq][row][col])
            else:
                print(training_set[sq][row][col])
                by=np.append(by,training_set[sq][row][col])
        plt.scatter(bx, by, label= "ball", color= "black",	marker= "o", s=20)
    
    # x-axis label 
        plt.title('Player Position') 
        plt.ylim(-34,34)
        plt.xlim(-52.5,52.5)
        plt.legend() 
        plt.show() 
        
        #save plot as image
        filename = sq + str('_') + str('r') + str(row) + str('.jpg')
        fig.savefig(filename)
    sq = 'sequence_'

