'''import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np                                         #scaing vs normalization
from scipy import stats
from sklearn.model_selection import train_test_split
# generate non-normal data
original_data = np.array([10,9,8,7,6,5,4,3,2,1])
print(original_data)
# split into testing & training data
train,test = train_test_split(original_data, shuffle=False)
print(train,test)
# transform training data & save lambda value
train_data,fitted_lambda = stats.boxcox(train)
print((train_data,fitted_lambda))
# use lambda value to transform test data
test_data = stats.boxcox(test, fitted_lambda)
print(test_data)
# (optional) plot train & test
fig, ax=plt.subplots(1,2)
sns.distplot(train_data, ax=ax[0])
sns.distplot(test_data, ax=ax[1])
plt.show()'''

#plotly exmaple::
import plotly.graph_objects as go
import plotly.offline as py
'''autosize =False
# Use `hole` to create a donut-like pie chart
values=[431000, 1520000, 294000]
labels=['Confirmed',"Recovered","Deaths"]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.update_traces(hoverinfo='label+percent', textinfo='value',textfont_size=15,
                  marker=dict(colors=['#00008b','#fffdd0'], line=dict(color='#FFFFFF', width=2.5)))
fig.update_layout(
    title='COVID-19 ACTIVE CASES VS CURED WORLDWIDE')
py.iplot(fig)'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("C:\\Users\\saketh\\Documents\\data.csv")
df=df.drop(['Unnamed: 32'],axis=1)
#print(df.head(10))
#print(df.columns)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in list(df.columns):
    if df[i].dtype == 'object':
        df[i] = le.fit_transform(df[i])
y=df['diagnosis']
x=df.drop('diagnosis',axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred.round())
print(score)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
pred_1=lr.predict(x_test)
score_1=accuracy_score(y_test,pred_1)
print(score_1)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
pred_2=rfc.predict(x_test)
score_2=accuracy_score(y_test,pred_2)
print(score_2)

from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(x_train,y_train)
pred_3=gbc.predict(x_test)
score_3=accuracy_score(y_test,pred_3)
print(score_3)

import plotly.express as px
df=pd.read_csv("C:\\Users\\saketh\\Documents\\covid.csv")
print(df.tail())
print(df.head())
print(df.shape)
df=df[df['Confirmed']>0]
print(df.head())
print(df[df.Country=='Italy'])
#spread in countries
fig=px.choropleth(df,locations='Country',locationmode='country names',color='Confirmed'
                 ,animation_frame='Date')
fig.update_layout(title_text="Global spread of COVID-19")
fig.show()
#deaths in countries
fig=px.choropleth(df,locations="Country",locationmode='country names',color='Deaths',
                 animation_frame='Date')
fig.update_layout(title_text='Global Deaths because of COVID-19')
fig.show()

#global infection rate
countries=list(df["Country"].unique())
max_infection_rates=[]
for c in countries:
    max_infected=df[df.Country==c].Confirmed.diff().max()
    max_infection_rates.append(max_infected)

df_MIR=pd.DataFrame()
df_MIR["Country"]=countries
df_MIR['Max Infection Rate']=max_infection_rates
fig=px.bar(df_MIR,x='Country',y='Max Infection Rate',color='Country',title='Global Max infection Rate',
           log_y=True)
fig.show()
