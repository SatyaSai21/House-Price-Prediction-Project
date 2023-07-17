import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None
__price=None

def get_estimated_price(location,bhk,total_sqft,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x=np.zeros(len(__data_columns))
    x[0] = bhk
    x[1] = total_sqft
    x[2] = bath
    if loc_index>=0:
        x[loc_index] = 1

    price = (__model.predict([x])[0])
    return price

def get_locations():
    return __locations

def load_saved_artifacts():
    print("Loading Saved Artifacts..........Start")
    global __locations
    global __data_columns
    with open("C:/Users/Sathya Sai/MLproject/server/artifacts/columns.json",'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    global __model
    with open("C:/Users/Sathya Sai/MLproject/server/artifacts/bangalore_home_prices_model.pickle",'rb') as fi:
        __model = pickle.load(fi)
        print("Loading Saved Artifacts..........Done")
        #<-----------------------testing----------------------------------------------->


if __name__ == "__main__" :
    load_saved_artifacts()
    print(get_locations())
    #print(get_estimated_price('1st Phase JP Nagar', 3,1000, 3))
    #print(get_estimated_price('1st Phase JP Nagar', 2, 1000, 2))
    #print(get_estimated_price('6th Phase JP Nagar', 5,3000, 5))
