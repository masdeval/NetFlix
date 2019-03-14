import csv
import regex as re
import pandas as pd
from collections import defaultdict

def createTrainData():

    dir = "/home/christian/data_analyse/NetFlix/netflix-prize-data/"
    data = defaultdict(lambda : defaultdict(lambda : 0))
    path = [dir + "combined_data_1.txt"]#,dir+"combined_data_2.txt",dir+"combined_data_3.txt"]

    for file in path:
         with open(file, encoding='latin-1') as f:
          for line in f:
              aux = line.split(',')
              if (len(aux) == 1):# and re.match(r'^\d+:', aux[0])!= None): #found a movie
               movie = int(re.match(r'^\d+',aux[0]).group(0))
              else:
                #assert(len(aux)>2)
                data[movie][int(aux[0])] = int(aux[1]) # [movie][user] = rating


    resultFile = open("users_id.txt", "r")
    users = resultFile.readline().split(',') #user id
    resultFile.close()
    resultFile = open("data_1.csv", "a")
    movies = sorted(list(data.keys()))  # movies id

    # create a file MOVIE x USER
    # for x in range(1,1000): # only 1000 movies
    #     line = list()
    #     #line.append(rows[x]) # movie id
    #     for y in range(1,100000): # only 1000000 users
    #          # some lines will be all zeros as there are gaps in the users id
    #          line.append(data[int(movies[x])][int(users[y])])
    #     resultFile.write(','.join([str(v) for v in line])+'\n')


    # create file USER x MOVIE
    resultFile = open("data_1.csv","w")
    for x in range(1,100000): #only 100000 users
        line = list()
        #line.append(rows[x]) # movie id
        for y in range(1,1000): #only 1000 movies
            #some columns will be all zeros as there are gaps in the users id
            #the advantage is that we can seek the user using its id as the line number
            line.append(data[int(movies[y])][int(users[x])])
        resultFile.write(','.join([str(v) for v in line])+'\n')


    resultFile.close()

createTrainData()

