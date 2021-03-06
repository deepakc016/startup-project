#!/usr/bin/env python
# coding: utf-8

# # startup project profit prediction, putting it on heroku

# In[ ]:


import pandas as pd
import sklearn

df= pd.read_csv("startup.csv") 

df=df.drop(['State'], axis=1)


x= df.iloc[:,:-1]
y= df.iloc[:,-1:]

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest= train_test_split(x,y, train_size=.80, random_state=10)

from sklearn.linear_model import LinearRegression
model= LinearRegression()

model.fit(xtrain, ytrain)

ypred= model.predict(xtest)

#from sklearn.metrics import r2_score
#r2_score(ypred, ytest)


# In[ ]:


from flask import Flask, render_template, request

app= Flask(__name__)

@app.route('/')

def xyz():
    return render_template("startup.html")

@app.route('/detail', methods = ['get', 'post'])

def abc():
    if (request.method=="POST"):
        x=int(request.form['a'])
        y=int(request.form['b'])
        z=int(request.form['c'])
        
        inpt=[[x,y,z]]
        pred=model.predict(inpt)
        
        return render_template("startup.html", profit=pred)

if __name__=='__main__':
    app.run()
    

