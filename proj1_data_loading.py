import json # we need to use the JSON package to load the data, since the data is stored in JSON format
from collections import Counter
import re
import nltk
nltk.download('stopwords')

with open("proj1_data.json") as fp:
    data = json.load(fp)
    

# Now the data is loaded.
# It a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment 
#print(data)




training= data[0:10000] #the list contains elements from position 0 to position 9999, total 10000
validation = data[10000:11000] #the list contains elements from position 10000 to position 10999, total 1000
testing=data[11000:12000] # the list contains elements from position 11000 to position 11999, total 1000


hugelist=['']*250000

#print(hugelist)
#print(training[0])

#print("length of training "+str(len(training)))
#print("length of validation "+str(len(validation)))
#print("length of testing "+str(len(testing)))

#print(testing[999])
#print(training)
#print("\n")
# Example:
j =0

for pos in range(len(training)): #e.x. for x in range(6): 0 1 2 3 4 5

	data_point = training[pos] # select the first data point in the dataset
	#print(data_point) 
	#print(data_point["text"])
	#print(data_point["text"].lower())
	# Now we print all the information about this datapoint
	#for info_name, info_value in data_point.items():
	#    print(info_name + " : " + str(info_value))
	#
	#print("\n")
	lowerx=data_point['text'].lower()
	#withoutpunc=lowerx.translate(None, string.punctuation)

	data_point["text"]=lowerx.split()
	
	
	

	#print(data_point)
	#for info_name, info_value in data_point.items():
	#    print(info_name + " : " + str(info_value))
	#print("\n")

	#print(len(data_point["text"]))
	slen= len(data_point["text"])


	
	#s = "string. With. Punctuation?"
	#s = re.sub(r'[^\w\s]','',s)


	#print(slen)
	#print(data_point["text"][0]) #specific word in "text"
	for i in range(slen): #e.x. for x in range(6): 0 1 2 3 4 5
	    
	    #data_point["text"][i]=data_point["text"][i].translate(None, string.punctuation)

	    data_point["text"][i] = re.sub(r'[^\w\s]','',data_point["text"][i]) # removing the punctuation
	    #print(data_point["text"][i])

	    hugelist[i+j]=data_point["text"][i]

	j=j+slen

#print(hugelist)
#print(training[9999])
hugelist = filter(None, hugelist) # decrease the list size by removing empty spot
#print(hugelist)

stop_words = nltk.corpus.stopwords.words('english')
filt_list = [word for word in hugelist if word not in stop_words]


#print(filt_list)

c=Counter(filt_list).most_common(160)
#print(c)

most=['']*160 #mopst frequent 160 words
for i in range(160):
	most[i]=c[i][0]

#print(most[0])

#xcounts=[[0]*160]*10000
print(most)

n = 10000
m = 160
xcounts = [0] * n
for i in range(n):
    xcounts[i] = [0] * m




countfre = 0

for pos in range(len(training)): #e.x. for x in range(6): 0 1 2 3 4 5

	data_point = training[pos]
	#print(data_point)
	slen= len(data_point["text"])

	for i in range(160): 
		for j in range(slen):
			#print(data_point["text"][j])
			if most[i] == data_point["text"][j]:
				countfre+=1
				
		#	print(countfre)		
		xcounts[pos][i]=countfre
		countfre=0
#print(xcounts)
	#print(xcounts)
		#countfre=0








