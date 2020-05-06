import numpy as np
import pandas as pd
import math
import time
import pickle

training_set = pd.read_pickle('train_data.pkl')#STATS DATASET

sq = 'sequence_'

activelist = {"1" : np.array([0,1]), 
              "2" : np.array([2,3]), 
              "3" : np.array([4,5]), 
              "4" : np.array([6,7]), 
              "5" : np.array([8,9]), 
              "6" : np.array([10,11]), 
              "7" : np.array([12,13]), 
              "8" : np.array([14,15]), 
              "9" : np.array([16,17]), 
              "10" : np.array([18,19]), 
              "11" : np.array([20,21])
              }

nonactivelist = {"1" : np.array([0,1]),
                 "2" : np.array([2,3]), 
                 "3" : np.array([4,5]),
                 "4" : np.array([6,7]),
                 "5" : np.array([8,9]),
                 "6" : np.array([10,11]),
                 "7" : np.array([12,13]),
                 "8" : np.array([14,15]),
                 "9" : np.array([16,17]),
                 "10" : np.array([18,19]),
                 "11" : np.array([20,21]),
                 "12" : np.array([22,23]),
                 "13" : np.array([24,25]),
                 "14" : np.array([26,27]),
                 "15" : np.array([28,29]),
                 "16" : np.array([30,31]),
                 "17" : np.array([32,33]),
                 "18" : np.array([34,35]),
                 "19" : np.array([36,37]),
                 "20" : np.array([38,39]),
                 "21" : np.array([40,41]),
                 "22" : np.array([42,43])      
                 }

goal_position = [1.0,0.0]

np.seterr(divide='ignore', invalid='ignore')
start = time.time()
X_train = {}
#Making GK-Def as a active player
for player in range(1,2):#len(training_set)
    print("activeplayer:",player)
    playerX = activelist[str(player)][0]
    playerY = activelist[str(player)][1]
    closest = np.array([])
    for i in range(1,6):#len(training_set)
        print(player,i)
        sq = sq + str(i) 
        temp = np.array([])
        for row in range(0,len(training_set[sq])):
            t = np.array([])         
            active_player_curr_x, active_player_curr_y = training_set[sq][row][playerX:playerY+1]
            for nonactive in range(1,23): 
                if row == 0:
                    prev_x,prev_y = 0,0
                else:
                    prev_x,prev_y = training_set[sq][row-1][nonactivelist[str(nonactive)][0]:nonactivelist[str(nonactive)][1]+1]
                #t = np.array([])
                current_x, current_y = 0, 0
                current_x, current_y = training_set[sq][row][nonactivelist[str(nonactive)][0]:nonactivelist[str(nonactive)][1]+1]
                #print(current_x)
                #print(current_y)
                t = np.append(t, current_x)
                t = np.append(t, current_y)
                ##print(t)
                ball_positionX, ball_positionY = training_set[sq][row][44:46]
                #Calculate the velocity
                velocity_x = (current_x - prev_x)
                velocity_y = (current_y - prev_y)
                velocity = velocity_x**2 + velocity_y**2
                t = np.append(t, velocity)
                #Calculate the distance
                distance_btwn_activeplayer = math.sqrt((current_x - active_player_curr_x)**2 + (current_y - active_player_curr_y)**2 )
                closest = np.append(closest, distance_btwn_activeplayer)
                t = np.append(t, distance_btwn_activeplayer)
                #Angle btwn active player
                num = (current_x*active_player_curr_x) + (current_y*active_player_curr_y)
                denum1 = math.sqrt((current_x**2)+(current_y**2))
                denum2 = math.sqrt((active_player_curr_x**2)+(active_player_curr_y**2))
                denum = denum1 * denum2
                if denum == 0:
                    angle_btwn_activeplayer = math.cos(denum)
                else:
                    X = num/denum
                    angle_btwn_activeplayer = math.cos(X)
                t = np.append(t, angle_btwn_activeplayer)         
                distance_btwn_goal = math.sqrt((current_x - goal_position[0])**2 + (current_y - goal_position[1])**2 )
                t = np.append(t, distance_btwn_goal)
                #Angle btwn goal 
                num = (current_x * goal_position[0]) + (current_y * goal_position[1])
                denum1 = math.sqrt((current_x**2)+(current_y**2))
                denum2 = math.sqrt((goal_position[0]**2)+(goal_position[1]**2))
                denum = denum1 * denum2
                denum = denum1 * denum2
                if denum == 0:
                    angle_btwn_goal = math.cos(denum)
                else:
                    X = num/denum
                    angle_btwn_goal = math.cos(X)
                t = np.append(t, angle_btwn_goal)
                distance_btwn_ball = math.sqrt((current_x - ball_positionX)**2 + (current_y - ball_positionY)**2 )
                t = np.append(t, distance_btwn_ball)
                #Angle btwn ball 
                num = (current_x * ball_positionX) + (current_y * ball_positionY)   
                denum1 = math.sqrt((current_x**2)+(current_y**2))
                denum2 = math.sqrt((ball_positionY**2)+(ball_positionY**2))
                denum = denum1 * denum2
                if denum == 0:
                    angle_btwn_ball = math.cos(denum)
                else:
                    X = num/denum
                    angle_btwn_ball = math.cos(X)
                t = np.append(t, angle_btwn_ball)
                #temp = np.append(temp, t)
            #Finding Closest Indices
            closest = closest[0:11]
            closest_index = closest.argsort()[:4]
            closestlist = np.array([])
            for k in closest_index:
                if closest[k]!=0:
                    ind = k
                    closestlist = np.append(closestlist, t[ind*9:(ind+1)*9])
            t = np.append(t, closestlist)
            temp = np.append(temp,t)
        temp = temp.reshape(len(training_set[sq]),390)
        X_train.update({(str(player),sq):temp})
        sq = 'sequence_'
end = time.time()
print('The execution time',end-start)
   
file = open('xtrain.pkl','wb')
pickle.dump(X_train,file)
file.close() 


