import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#Fetch data and format it as needed
data=fetch_movielens(min_rating=4.0)

#display

#create a model

model=LightFM(loss='warp')#weighted approx rank otherwise
#Uses gradient descent
#now train model
model.fit(data['train'],epochs=750,num_threads=2)

def sample_recommendation(model,data,user_ids):
  #no. of users and movies
  n_users,n_items=data['train'].shape
  #Generate recommendations for each
  for user_id in user_ids:
    #Movies they already like
    known_positives=data['item_labels'][data['train'].tocsr()[user_id].indices]
    #Movies they may like as per the model
    scores=model.predict(user_id,np.arange(n_items))
    #rank them from most liked to least
    top_items=data['item_labels'][np.argsort(-scores)]
    
    #print results
    print('User Id:'+str(user_id))
    print("Known Positives:")
    print("-----------------")
    for x in known_positives[:3]:
      print(str(x))
    print("")
    print("Recommended:")
    print("-----------------")
    for x in top_items[:3]:
      print(str(x))
    print("")
    print("x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x")
    
#Call the function

sample_recommendation(model,data,[1,2,3,25,45,4,99])