import json # we need to use the JSON package to load the data, since the data is stored in JSON format
from collections import Counter
import re
import numpy as np
import operator
import math

from implement_linreg import gd_lin_reg, cf_lin_reg
from implement_linreg import compute_mse

import matplotlib.pyplot as plt
import time
start_time = time.time()






with open("proj1_data.json") as fp:
    data = json.load(fp)
    



training= data[0:10000] #the list contains elements from position 0 to position 9999, total 10000
validation = data[10000:11000] #the list contains elements from position 10000 to position 10999, total 1000
testing=data[11000:12000] # the list contains elements from position 11000 to position 11999, total 1000



trainlen=len(training)

hugelist=['']*500000

#children=[0]*trainlen
#childrensquare=[0]*trainlen
#controversiality=[0]*trainlen
#is_root=[0]*trainlen
#popularity_score=[0]*trainlen
k =0
for pos in range(trainlen): #e.x. for x in range(6): 0 1 2 3 4 5


  data_point1 = training[pos] # select the first data point in the dataset
  
  #children[pos]=data_point['children']
  #controversiality[pos]=data_point['controversiality']
  #is_root[pos]=int(data_point['is_root'])
  #childrensquare[pos]=data_point['children']*data_point['children']
  #popularity_score[pos]=data_point['popularity_score']
  #print(data_point["text"])
  #print(data_point["text"].lower())
  # Now we print all the information about this datapoint
  #for info_name, info_value in data_point.items():
  #    print(info_name + " : " + str(info_value))
  #
  #print("\n")
  lowerx=data_point1['text'].lower()
  #withoutpunc=lowerx.translate(None, string.punctuation)

  data_point1["text"]=lowerx.split()
  #print(data_point)
  #for info_name, info_value in data_point.items():
  #    print(info_name + " : " + str(info_value))
  #print("\n")

  #print(len(data_point["text"]))
  slen= len(data_point1["text"])

  
  #s = "string. With. Punctuation?"
  #s = re.sub(r'[^\w\s]','',s)


  #print(slen)
  #print(data_point["text"][1]) #specific word in "text"
 
  for i in range(slen): #e.x. for x in range(6): 0 1 2 3 4 5
      
      #data_point["text"][i]=data_point["text"][i].translate(None, string.punctuation)

      #data_point["text"][i] = re.sub(r'[^\w\s]','',data_point["text"][i]) # removing the punctuation
      #print(data_point["text"][i])

      hugelist[i+k]=data_point1["text"][i]

  k=k+slen

#print(hugelist)
#print(training[9999])

#print popularity_score

hugelist = filter(None, hugelist) # decrease the list size by removing empty spot
#print(hugelist)
#print("is_root", is_root)
#print(filt_list)

c=Counter(hugelist).most_common(60) #most frequent 160 words ********************
#print(c)

most=['']*60 #most frequent 160 words ********************
for i in range(60):
  most[i]=c[i][0]


#print(most)
#-------------------------------------------------------------------------------
def transform(partition):

  print(most)
  partlen=len(partition)
  hugelist=['']*250000
  j =0
  children=[0]*partlen
  childrensquare=[0]*partlen
  controversiality=[0]*partlen
  is_root=[0]*partlen
  popularity_score=[0]*partlen

  for pos in range(partlen): #e.x. for x in range(6): 0 1 2 3 4 5


    data_point = partition[pos] # select the first data point in the dataset
    
    children[pos]=data_point['children']
    controversiality[pos]=data_point['controversiality']
    is_root[pos]=int(data_point['is_root'])
    #childrensquare[pos]=data_point['children']*data_point['children']
    popularity_score[pos]=data_point['popularity_score']

  n = len(partition)
  m = 64
  xcounts = [0] * n
  for i in range(n):
      xcounts[i] = [0] * m

  countfre = 0
  



  for pos in range(partlen): #e.x. for x in range(6): 0 1 2 3 4 5

    data_point = partition[pos]
    



    lowerx=data_point['text'].lower()
    #withoutpunc=lowerx.translate(None, string.punctuation)

    data_point["text"]=lowerx.split()




    slen= len(data_point["text"])


    #for pos in range(partlen):
    #     print(most[0])
    print(data_point)
    print("fucccccccck",data_point["text"][0])

    for i in range(60): 
      for j in range(slen):
        #print(data_point["text"][j])
        print(most[i])
        print(data_point["text"][j])
        if most[i] == data_point["text"][j]:

          countfre+=1
          
      # print(countfre)   
      xcounts[pos][i]=countfre
      countfre=0
    
    #print(xcounts)

    xcounts[pos][60] = children[pos]
    xcounts[pos][61] = controversiality[pos]
    xcounts[pos][62] = is_root[pos]
    xcounts[pos][63] = 1  #bias term
  #xcounts[pos][64] = childrensquare[pos]
  
  #if children[pos]!=0:
  # xcounts[pos][65] = math.log(children[pos])*(1-controversiality[pos])
  #else:
  # xcounts[pos][65] = 0

  #print(most[0])
  #print(xcounts)
    #countfre=0


  X = np.asarray(xcounts)
  Y = np.asarray(popularity_score)
  print(X)
#print(X.shape)
  print("caonimabi")
  return X,Y


#transform(training)
#X_train,y_train = transform(training)
#print(X_train)
#X_train,y_train = transform(training)#input training/validation/testing
X_val,y_val = transform(validation)
for x in range(1000):
  print(X_val[x][0])
#X_test,y_test = transform(testing)


#------------------------------------------------

# Implementing linear regression using closed-form solution:
w_cf = cf_lin_reg(X_train, y_train)
#y_predict = X_train @ w_cf # y_predict is [10000*1], X_train is [10000*165], w_cf is [165*1]

# Calulating mean squared error
MSE_cf_train = compute_mse(X_val, w_cf, y_val)
print("MSE: ", MSE_cf_train)

# Plotting training targets vs predicted values
#plt.plot(y_train, y_predict, 'bo', markersize=0.5)
#plt.xlabel = 'y_train'
#plt.ylabel = 'y_predict'
#plt.xlim(-7.5, 10)
#plt.ylim(-2.5, 10)
#plt.gca().set_aspect('equal', adjustable='box')
#plt.show()
#plt.clf()
print("--- %s seconds ---" % (time.time() - start_time))




