#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import seaborn as sns


# In[3]:



def split_data(source,ratio):
    data=load_data(source)
    
    


# In[2]:


source="C:\\Users\\Sathya Sai\\Downloads\\Bengaluru_House_Data.csv"
#f=load_data(source)


# In[3]:


def load_data(source,split_ratio):
    data=[]
    with open(source,'r+') as file:
        data=file.readlines()
    dictlist=[]
    header=parse_header(data[0])
    for i in data[1:]:
        col=parse_values(i)
        dictionary=create_dict(header,col)
        dictlist.append(dictionary)
    
        
        
        
    


# In[4]:


def parse_values(data):
    d=[item for item in data.strip().split(',')]
    return d
def create_dict(header,values):
    dictionary={}
    for i,j in zip(header,values):
        dictionary[i]=j
    return dictionary
def parse_header(headers):
    head=[feature for feature in headers.strip().split(',')]
    return head


# In[5]:


data=pd.read_csv(source,header=0)
data


# # MEMORY_USAGE:

# In[6]:


data.info()


# In[7]:


total_memory=0
for i in data.memory_usage(deep=True):
    total_memory+=i
print("DATA IN MB = "+str(total_memory/(1024*1024)))   


# In[8]:


data.memory_usage(deep=True)  #in bytes


# In[9]:


data.head()


# In[10]:


data_b=data.drop(['area_type','availability','balcony','society'],axis='columns')


# In[11]:


data_b.head()


# # DATA CLEANING

# In[12]:


data_b.isna().any()#---->gives if any columns have NaN values


# In[13]:


data_b.isnull().sum()#--->No. Of NaN Values in each column


# In[14]:


#AS LOCATION HAS ONLY ONE NAN VALUE WE CAN DROP THAT COLUMN
data_b=data_b[data_b["location"].notnull()]


# In[15]:


#ALSO ALL Columns ARE NOT MISSING AT RANDOM THEREFORE DROPPING THE NaN Values
data_b=data_b.dropna()


# In[16]:


data_b.isnull().sum()


# In[17]:


data['bath'].value_counts()


# In[18]:


plt.figure(figsize=(10,5))
sns.histplot(data["bath"])
plt.xlabel('NUMBER OF BATHROOMS')
plt.ylabel('COUNT')


# In[19]:


data_b


# In[20]:


data_b["size"]=data_b["size"].apply(lambda x: int(x.split(' ')[0]))


# In[21]:


data_b.sample(n=5,random_state=4)


# In[22]:


data_b.rename(columns={'size':'bhk'},inplace=True)


# In[23]:


data_b#["N_BHK"].unique()


# In[24]:


data_b[data_b.bhk==43] #43 bedrooms in a space of 2400 sqft !LOL  !!!!!!OUTLIER


# In[25]:


data_b[data_b.bhk==27]


# In[26]:


data_b["total_sqft"].unique()#---->'1133 - 1384'error need to be handled


# In[27]:


#data_b.total_sqft.dtype#---->OBJECT
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
t=data_b.head()


# In[28]:


temp=data_b[~data_b["total_sqft"].apply(is_float)]
temp
#we observe some ranges therefore i am just averaging the range and replace it instead of that data


# In[29]:


p=data_b.copy()


# In[33]:


p["total_sqft"]=p["total_sqft"].apply(helper)
p.isna().sum()


# In[34]:


index=p[p["total_sqft"].isnull()].index
index


# In[35]:


data_b.loc[12186,:]#------------------>after cleaning the data before it is like xxxSq. Meter etc...


# #Object Datatype Handling

# In[36]:


def stringcutter(s):
    i=len(s)-1
    while (((s[i]>='A') and (s[i]<='Z')) or ((s[i]>='a') and (s[i]<='z')) or s[i]=='.' or s[i]==' ') and i>=0:
        i-=1
    l=pd.to_numeric(s[0:i+1])
    b=s[i+1:]
    dict={'Guntha':np.NaN,'Sq. Meter':l*10.7639,'Sq. Yards':l*9,'Acres':np.NaN,'Grounds':np.NaN,'Cents':np.NaN,'Perch':np.NaN,'':l}
    return np.round(dict[b],2)
    


# In[37]:


data_b.loc[index,"total_sqft"]=data_b.loc[index,"total_sqft"].apply(stringcutter)#---->object


# In[38]:


r=stringcutter('600Sq. Meter')
print(r) #==10.7639*600


# In[39]:


data_b.head()


# In[40]:


#we observe some ranges therefore i am just averaging the range and replace it instead of that data
def helper(d):
    if(is_float(d)!=True):
        temp=d.strip().split('-')
        if(len(temp)==2):
            return (float(temp[0])+float(temp[1]))/2
    else:
        return float(d)
    try:
        return float(x)
    except:
        return np.nan


# In[41]:


data_b["total_sqft"]=data_b["total_sqft"].apply(helper)


# In[42]:


data_b.loc[index,"total_sqft"][0:4]


# In[43]:


data_b.dropna(inplace=True) #after data handling the 14 nan values present are removed 


# In[44]:


data_b.isna().sum()


# In[45]:


data_b.sample(n=5,random_state=1)


# In[46]:


data_b["price_per_sqft"]=(data_b["price"]*100000)/data_b["total_sqft"]


# In[47]:


data_b.head()


# #analysing location column

# In[48]:


res=data_b["location"].value_counts()<=10
res


# In[49]:


len(data_b["location"].unique()) ## THIS CREATES DIMENSIONALITY PROBLEM I.E.,DIMENSIONALITY CURSE


# In[50]:


data_b["location"].value_counts()


# In[51]:


def oo(x):
    if(res[x]==False):
        return x
    else:
        return "Others"


# In[52]:


#oo('Whitefield')
#print(oo('Whitefield'))-------------->data_b["location"].value_counts()['Whitefield']
#oo('Kannur')
#print(oo('Kannur'))---------------------->data_b["location"].value_counts()['Kannur']


# In[53]:


data_b["location"]=data_b["location"].apply(oo)


# In[54]:


print("NUMBER OF UNIQUE VALUES AFTER DATA CLEANING  : {}".format(len(data_b["location"].unique())))
#------------------->now the Number of dimensions reduced a lot and 
#--------------------------------------------------->hence solving the problem of dimensionality curse


# In[55]:


#OUTLIER REMOVAL


# In[56]:


#average area of one bed room 558.47


# In[58]:


data_b=data_b[~((data_b["total_sqft"]/data_b["bhk"])<300)]


# In[59]:


data_b["price_per_sqft"].describe()


# In[60]:


def remove_location_outliers(df):
    data_t=pd.DataFrame()
    for key,subdataset in df.groupby('location'):
        m=np.mean(subdataset["price_per_sqft"])
        s=np.std(subdataset["price_per_sqft"])
        reduced=subdataset[(subdataset["price_per_sqft"]<=(m+s)) & (subdataset["price_per_sqft"]>=(m-s))]
        data_t=pd.concat([data_t,reduced],axis=0,ignore_index=True)
    return data_t


# In[61]:


data_b=remove_location_outliers(data_b)
len(data_b)


# In[62]:


data_b.head()


# In[63]:


'''reduced=pd.DataFrame
reduced=f(data_b)
len(reduced)
#np.mean(reduced["price_per_sqft"])'''


# In[64]:


data_b["price_per_sqft"].describe()


# In[65]:


q75=data_b["price_per_sqft"].quantile(0.75)


# In[66]:


q25=data_b["price_per_sqft"].quantile(0.25)


# In[67]:


IQR=q75-q25
IQR


# In[68]:


qmin=q25-1.5*IQR
qmin


# In[69]:


qmax=q75+1.5*IQR
qmax


# In[70]:


temp=data_b[(data_b["price_per_sqft"]<=qmax) &  (data_b["price_per_sqft"]>=qmin)]


# In[71]:


len(data_b[(data_b["price_per_sqft"]<=qmax) &  (data_b["price_per_sqft"]>=qmin)])


# In[72]:


temp["price_per_sqft"].describe()


# In[73]:


sns.boxplot(data=temp["price_per_sqft"])


# In[76]:


def scatter_plot(df,location):
    bhk2=df[(df['location']==location) & (df["bhk"]==2)]
    bhk3=df[(df['location']==location) & (df["bhk"]==3)]
    #matplotlib.rcparams['figure.figsize']=(15,10)
    plt.scatter(bhk2["total_sqft"],bhk2["price_per_sqft"],c='g',marker='+',s=50,label='2_BHK')
    plt.scatter(bhk3["total_sqft"],bhk3["price_per_sqft"],c='r',marker='*',s=50,label='3_BHK')
    plt.title(location)
    plt.xlabel('Total Sqft Area')
    plt.ylabel('Price Per Sqft')
    plt.legend()
    plt.show()


# In[88]:


for i in data_b["location"].unique():
    scatter_plot(data_b,i)


# In[77]:


scatter_plot(data_b,location=' Devarachikkanahalli') #TESTING


# In[78]:


p="Deva"
bhk2=data_b[data_b.location == 'Rajaji Nagar']# & (data_b["N_BHK"] == 2)]    
bhk3=data_b[(data_b['location']=='Rajaji Nagar') & (data_b["bhk"]==3)]
#rcparams['figure.figsize']=(15,10)
print(bhk2[0:3])
print("\n")
print(bhk3[0:3])


# In[79]:


plt.scatter(bhk2["total_sqft"],bhk2["price_per_sqft"],c='g',marker='+',s=50,label='2_BHK')
plt.scatter(bhk3["total_sqft"],bhk3["price_per_sqft"],c='r',marker='*',s=50,label='3_BHK')
#plt.title(location)
plt.xlabel('Total Sqft Area')
plt.ylabel('Price Per Sqft')
plt.legend()
plt.show()


# In[80]:


data_b[data_b["location"]==' Devarachikkanahalli']


# In[81]:


len(data_b["location"].unique())


# In[84]:


def remove_outliers_in_bhk(df):
    exclude_indices=[]
    for i,subd in df.groupby('location'):
        dict_b={}
        for j,bhk_sub in subd.groupby('bhk'):
            dict_b[j]={
                'count':bhk_sub.shape[0],
                'mean_price':np.mean(bhk_sub["price_per_sqft"]),
                'std':bhk_sub["price_per_sqft"].std()
                }
        for j,subdata in subd.groupby('bhk'):
            bhk=dict_b.get(j-1)#we are deleting from bhk==2 or more
            if ((bhk!=None) and (bhk['count']>5)):
                exclude_indices = np.append(exclude_indices,subdata[subdata["price_per_sqft"]<bhk['mean_price']].index)
    return df.drop(exclude_indices,axis=0)    


# In[85]:


d=remove_outliers_in_bhk(data_b)


# In[96]:


####------------------->TESTING
exclude_indices=[]
for i,subd in data_b.groupby('location'):
    dict_b={}
    for j,bhk_sub in subd.groupby('N_BHK'):
        dict_b[j]={
            'count':bhk_sub.shape[0],
            'mean_price':np.mean(bhk_sub["price_per_sqft"]),
            'std':bhk_sub["price_per_sqft"].std()
        }
    for j,subdata in subd.groupby('N_BHK'):
        bhk=dict_b.get(j-1)#we are deleting from bhk==2 or more
        if ((bhk!=None) and (bhk['count']>=5)):
            exclude_indices = np.append(exclude_indices,subdata[subdata["price_per_sqft"]<bhk['mean_price']].index)
    print(exclude_indices)
    break


# In[86]:


d


# # COMPARISON BEFORE AND AFTER REMOVING OUTLIERS

# In[87]:


scatter_plot(data_b,location=' Devarachikkanahalli') #TESTING
scatter_plot(d,' Devarachikkanahalli')


# In[88]:


data_b=d


# In[89]:


data_b


# In[90]:


plt.hist(data_b["price_per_sqft"],bins=10,rwidth=0.95) #inference----->right-skewed


# In[91]:


plt.hist(np.log(data_b["price_per_sqft"]),bins=10,rwidth=0.95) #transforming into NORMAL Distribution


# In[92]:


plt.hist(data_b["bath"],bins=10,rwidth=0.95)


# In[93]:


data_b["bath"].unique() #having 13,16 bathrooms looks unobvious


# In[94]:


data_b[(data_b["bath"]>=10) & (data_b["bath"]<=16)]
#12 bathrooms in 4000 sqft is a flaw.
#we need to handle these.


# In[95]:


def outlier_bath_rooms(df):
    return df[df["bath"]<(df["bhk"]+2)]
#1 bathroom for each bedroom and 1 extra atmost


# In[96]:


data_b=outlier_bath_rooms(data_b)


# In[97]:


len(data_b)


# In[98]:


data_b[data_b["bath"]>(data_b["bhk"]+2)] #Hence Cleaned.


# In[101]:


temp=pd.get_dummies(data_b["location"],drop_first=True)


# In[104]:


data_b=pd.concat([data_b,temp],axis=1)


# In[ ]:





# In[110]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder


# In[111]:


le=LabelEncoder()
oe=OneHotEncoder()


# In[112]:


b=pd.DataFrame()


# In[113]:


b["t"]=data_b["location"]


# In[112]:


b["t"]=pd.DataFrame(le.fit_transform(data_b["location"]))


# In[118]:


b.index#b=oe.fit_transform(data_b["location"]).toarray()


# In[106]:


oe.fit(b)#.categories_


# In[114]:


k=pd.DataFrame(oe.fit_transform(b).toarray(),index=b.index)


# In[115]:


l=oe.categories_


# In[116]:


l[0][24]


# In[117]:


for i in range(241):
    k.rename(columns={i:l[0][i]},inplace=True)


# In[118]:


data_b=pd.concat([data_b,k],axis=1)


# In[119]:


k.index


# In[105]:


t=data_b.drop(['price_per_sqft'],axis=1,inplace=True)

##-------------------------> Dropping one column from to avoid MULTICOLLINEARITY Problem  <--------------------------##


# In[106]:


data_b.head(n=5)


# In[ ]:


@app.route('/predict_home_price',methods=['POST'])
def predict_home_price():
    total_sqft=float(request.form['total_sqft'])
    location=request.form['location']
    N_BHK=int(request.form['N_BHK'])
    bath= int(request.form['N_BHK'])

    response=jsonify({
        'estimated_price':util.get_estimated_price(location,N_BHK,total_sqft,bath)
    })

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


# In[107]:


y=data_b["price"]
x=data_b.drop('price',axis=1)


# In[108]:


x=x.drop('location',axis=1)


# In[109]:


from sklearn.model_selection import train_test_split


# In[110]:


x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.2,random_state=20)


# In[111]:


print("total size: {}".format(len(data_b)))
print("train size: {}".format(len(x_train)))
print("test size: {} ".format(len(y_test)))


# In[127]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()


# In[128]:


model.fit(x_train,x_test)


# In[129]:


model.score(y_train,y_test)


# In[115]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=10)
cross_val_score(LinearRegression(),x,y,cv=cv)


# In[116]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[117]:


def finding_best_model(x,y):
    Algorithms={
        'DecisionTree_Regressor':{
            'model' : DecisionTreeRegressor(),
            'params' : {
                'criterion' :['squared_error','friedman_mse'],#,'poisson'],
                'splitter' :['random','best'],
                'random_state':[None,10],
                'max_features':['auto','sqrt','log2']
            }
        },
        'XGB_Regressor':{
            'model':XGBRegressor(),
            'params':{
                'loss':['squared_error','huber','quantile'],
                'criterion':['friedman_mse']
                
            }
        },
        'Lasso_regression':{
            'model':Lasso(),
            'params':{
                'selection':['cyclic', 'random'],
                'alpha':[1,2,3],
                'tol':[0.0004]
            }
        },
        'Ridge_regression':{
            
            'model':Ridge(),
            'params':{
                'solver':['auto','svd','cholesky','lsqr','sparse_cg','sag','saga'],
                'alpha':[1,2]
            }
        },
        'RandomForest_Regressor':{
            'model':RandomForestRegressor(),
            'params':{
                'n_estimators':[100,125,150],
                'criterion':['squared_error','friedman_mse'],
                'min_samples_split':[2,4]
                
            }   
        },
        'linear_regression' : {
            
            'model': LinearRegression(),##Pipeline([('preprocessor',StandardScaler()),('estimator',LinearRegression())]),#Pipeline([(StandardScaler),(LinearRegression()),
            'params':{}
        }
    }
    bs=" "
    scores=[]
    #df=pd.DataFrame()
    cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=10)
    for algo,parameters in Algorithms.items() :
        gcv=GridSearchCV(parameters['model'],parameters['params'],cv=cv,return_train_score=False)
        gcv.fit(x,y)
        scores.append({
            'model_name':algo,
            'best_score':gcv.best_score_,
            'best_params':gcv.best_params_
        })
        bs=gcv.best_estimator_
        #df=pd.concat([df,gcv.cv_results_],axis=0)
    print("Best_Estimator : {}".format(bs))
    return pd.DataFrame(scores,columns=['model_name','best_score','best_params'])


# In[118]:


d_b=finding_best_model(x_train,x_test)


# In[ ]:


b=


# In[184]:


d_b


# In[174]:


a= DecisionTreeRegressor(criterion='friedman_mse', max_features='sqrt', random_state=10, splitter= 'best')


# In[175]:


a.fit(x_train,x_test)


# In[177]:


a.score(y_train,y_test)


# #######################----BUT RANDOMFOREST IS COSTLY AND HOWEVER THERE ISN'T MUCH IMPROVEMENT IN SCORE WE ARE GOING WITH MULTIVARIATE REGRESSION ONLY----------#################

# In[191]:


def predict_target(location,sqft,bath,bhk):
    loc_index=np.where(x.columns==location)[0][0]
    p=np.zeros(len(x.columns))
    p[0]=bhk
    p[1]=sqft
    p[2]=bath
    if loc_index >= 0:
        p[loc_index]=1
    print(model.predict([p])[0])


# In[192]:


predict_target('1st Phase JP Nagar',1000,2,2)


# In[ ]:





# In[ ]:


Best_Estimator : DecisionTreeRegressor({'criterion': 'friedman_mse', 'max_features': 'sqrt', 'random_state': 10, 'splitter': 'best'})#---------->(0.8304752948882247)


Best_Estimator : XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, criterion='friedman_mse',
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=None, gpu_id=None,
             grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=None,
             loss='squared_error', max_bin=None, max_cat_threshold=None,
             max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
             max_leaves=None, min_child_weight=None, missing=nan,
             monotone_constraints=None, n_estimators=100, n_jobs=None,
             num_parallel_tree=None, ...)#------------------------------------>(0.8816423807527688)

Best_Estimator : Lasso(alpha=3, selection='random', tol=0.0004)#------------>(0.6535273594270129)
    
Best_Estimator : Ridge(alpha=1)#------------->(0.8074377823144514)

Best_Estimator : LinearRegression()#--------->(0.8032747864642041)

Best_Estimator : RandomForestRegressor(criterion='friedman_mse', min_samples_split=4,
                      n_estimators=150)#---------------->(0.8523184221566867)


# In[ ]:



        'DecisionTree_Regressor':{
            'model' : DecisionTreeRegressor(),
            'params' : {
                'criterion' :['squared_error','friedman_mse'],#,'poisson'],
                'splitter' :['random','best'],
                'random_state':[None,10],
                'max_features':['auto','sqrt','log2']
            }
        },
        
        'XGB_Regressor':{
            'model':XGBRegressor(),
            'params':{
                'loss':['squared_error','huber','quantile'],
                'criterion':['friedman_mse']
                
            }
        },
       
        'Lasso_regression':{
            'model':Lasso(),
            'params':{
                'selection':['cyclic', 'random'],
                'alpha':[1,2,3],
                'tol':[0.0004]
            }
        },
        'Ridge_regression':{
            
            'model':Ridge(),
            'params':{
                'solver':['auto','svd','cholesky','lsqr','sparse_cg','sag','saga'],
                'alpha':[1,2]
            }
        },
        
         'RandomForest_Regressor':{
            'model':RandomForestRegressor(),
            'params':{
                'n_estimators':[100,125,150],
                'criterion':['squared_error','friedman_mse'],
                'min_samples_split':[2,4]
                
            }
            
        },
        
            'linear_regression' : {
            'model': LinearRegression(),##Pipeline([('preprocessor',StandardScaler()),('estimator',LinearRegression())]),#Pipeline([(StandardScaler),(LinearRegression()),
            'params':{}
        },
                


# In[152]:


d=Pipeline([('preprocessor',StandardScaler()),('estimator',LinearRegression())])


# In[178]:


def predict_target(location,sqft,bath,bhk):
    loc_index=np.where(x.columns==location)[0][0]
    p=np.zeros(len(x.columns))
    p[0]=bhk
    p[1]=sqft
    p[2]=bath
    if loc_index >= 0:
        p[loc_index]=1
    print(model.predict([p])[0])


# In[150]:


a=np.where(x.columns==' Devarachikkanahalli')[0]


# In[151]:





# In[ ]:





# In[158]:


a


# In[183]:


predict_target('Indira Nagar',1000,2,2)


# In[193]:


import pickle
import json


# In[197]:


with open('bangalore_home_prices_model.pickle','wb') as f:
    pickle.dump(model,f)


# In[198]:


columns={
    'data_columns':[col.lower() for col in x.columns]
}
with open('columns.json','w') as file:
    file.write(json.dumps(columns))


# In[196]:


len(columns['data_columns'])


# In[ ]:




