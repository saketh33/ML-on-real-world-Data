import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

a='black dimgray dimgrey gray darkgray darkgrey silver lightgray lightgrey gainsboro whitesmoke white snow';black_white=str(a).split()
b=' rosybrown lightcoral indianred brown firebrick maroon darkred red mistyrose salmon tomato darksalmon coral' \
  ' orangered lightsalmon sienna seashell chocolate saddlebrown sandybrown peachpuff peru ';brown_red=str(b).split()
c='bisque darkorange burlywood navajowhite blanchedalmond papayawhip moccasin orange wheat oldlace floralwhite darkgoldenrod goldenrod cornsilk ' \
  'khaki palegoldenrod darkkhaki Ivory beige lightyellow lightgoldenrodyellow ';biscuit_gold=str(c).split()
d='olive yellow olivedrab yellowgreen darkolivegreen greenyellow chartreuse lawngreen honeydew darkseagreen palegreen lightgreen forestgreen limegreen darkgreen' \
  ' lime seagreen mediumseagreen springgreen mintcream ';yellow_green=str(d).split()
e='aquamarine turquoise lightseagreen mediumturquoise azure ' \
  'lightcyan paleturquoise darkslategray darkslategrey darkcyan aqua cadetblue powderblue lightskyblue steelblue dodgerblue ';blue=str(e).split()
f='lightslategray lightsteelblue cornflowerblue royalblue ghostwhite lavender ' \
  'midnightblue navy darkblue mediumblue blue slateblue darkslateblue mediumslateblue mediumpurple rebeccapurple ' \
  'blueviolet indigo darkorchid darkviolet plum violet purple darkmagenta magenta orchid mediumvioletred ' \
  'deeppink hotpink palevioletred crimson pink lightpink';blue_pink=str(f).split()

cars=pd.read_csv("C:\\Users\\saketh\\Documents\\carprice assignment.csv")
print(cars.head(10))
print(cars.shape)
print(cars.describe())
print(cars.info())
print(cars.isnull().sum())
print(cars.isnull().any())
cars.columns = map(str.lower, cars.columns)

cars.rename(columns={'carname':'companyname'},inplace=True)
cars['companyname']=cars['companyname'].apply(lambda x:x.split()[0])
#print(cars.companyname.unique())
def carname(a,b):
    cars['companyname'].replace(a,b,inplace=True)
carname('maxda','mazda')
carname('porcshce','porsche')
carname('toyouta','toyota')
carname('vokswageen','volkswagen')
carname('vw','volkswagen')
carname('Nissan','nissan')
print(cars.companyname.unique())
print(cars.price.describe(percentiles = [0.25,0.50,0.75,0.85,0.90,1]))

sns.displot(cars.price,kde=True,color='y')
plt.title('Car Price Spread')
plt.show()
sns.boxplot(y=cars.price)
plt.show()

plt.figure(figsize=(25, 6))
plt.subplot(1,3,1)
plt1 = cars.companyname.value_counts().plot(kind='bar',color=blue_pink)
plt.title('Companies Histogram')
plt1.set(xlabel = 'Car company', ylabel='Frequency of company')
plt.subplot(1,3,2)
plt1 = cars.fueltype.value_counts().plot(kind='bar',color=yellow_green)
plt.title('Fuel Type Histogram')
plt1.set(xlabel = 'Fuel Type', ylabel='Frequency of fuel type')
plt.subplot(1,3,3)
plt1 = cars.carbody.value_counts().plot(kind='bar',color=blue)
plt.title('Car Type Histogram')
plt1.set(xlabel = 'Car Type', ylabel='Frequency of Car type')
plt1=cars.companyname.value_counts().plot.bar()
plt.show()

#print(cars.companyname.value_counts())

df = pd.DataFrame(cars.groupby(['companyname'])['price'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Company Name vs Average Price')
plt.show()

#sns.countplot(cars.symboling, palette=("cubehelix"))
#sns.countplot(y=cars.companyname.value_counts(),palette=("cubehelix"))
plt1 = cars.companyname.value_counts().plot(kind='bar',color='r')
plt.title('Companies Histogram')
plt1.set(xlabel='Car company', ylabel='Frequency of company')
plt.show()

plt.subplot(1,2,1)
plt.title('Symboling Histogram')
sns.countplot(cars.symboling, palette=("cubehelix"))
plt.subplot(1,2,2)
plt.title('Symboling vs Price')
sns.boxplot(x=cars.symboling, y=cars.price, palette=("cubehelix"))
plt.show()

cars['fueleconomy'] = (0.55 * cars['citympg']) + (0.45 * cars['highwaympg'])
cars['price'] = cars['price'].astype('int')
temp = cars.copy()
table = temp.groupby(['companyname'])['price'].mean()
temp = temp.merge(table.reset_index(), how='left',on='companyname')
bins = [0,10000,20000,40000]
cars_bin=['Budget','Medium','Highend']
cars['carsrange'] = pd.cut(temp['price_y'],bins,right=False,labels=cars_bin)
cars_lr = cars[['price', 'fueltype', 'aspiration','carbody', 'drivewheel','wheelbase',
                  'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower',
                    'fueleconomy', 'carlength','carwidth', 'carsrange']]
def dummies(x, df):
  temp = pd.get_dummies(df[x], drop_first=True)
  df = pd.concat([df, temp], axis=1)
  df.drop([x], axis=1, inplace=True)
  return df
cars_lr = dummies('fueltype',cars_lr)
cars_lr = dummies('aspiration',cars_lr)
cars_lr = dummies('carbody',cars_lr)
cars_lr = dummies('drivewheel',cars_lr)
cars_lr = dummies('enginetype',cars_lr)
cars_lr = dummies('cylindernumber',cars_lr)
cars_lr = dummies('carsrange',cars_lr)
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(cars_lr, train_size = 0.7, test_size = 0.3, random_state = 100)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','fueleconomy','carlength','carwidth','price']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


print(df_train.head())
plt.figure(figsize = (30, 25))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()
