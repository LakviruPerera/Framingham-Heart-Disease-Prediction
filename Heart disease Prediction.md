```python
# Importing Packages
# Plotting
import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
%matplotlib inline
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
# calculate point-biserial correlation
import scipy.stats as stats
# Data Resampling
from sklearn.utils import resample
# Data Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# splitting data
from sklearn.model_selection import train_test_split
# Data Scaling
from sklearn.preprocessing import MinMaxScaler
# Data Modeling
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, classification_report
```


```python
# Importing dataset
df = pd.read_csv('C:/Users/Lakviru Perera/Desktop/4th YEAR/IS 4007 Statistics in Practice II/Frahmingham/framingham.csv')                    
```


```python
df.shape
```




    (4238, 16)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4238 entries, 0 to 4237
    Data columns (total 16 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   male             4238 non-null   int64  
     1   age              4238 non-null   int64  
     2   education        4133 non-null   float64
     3   currentSmoker    4238 non-null   int64  
     4   cigsPerDay       4209 non-null   float64
     5   BPMeds           4185 non-null   float64
     6   prevalentStroke  4238 non-null   int64  
     7   prevalentHyp     4238 non-null   int64  
     8   diabetes         4238 non-null   int64  
     9   totChol          4188 non-null   float64
     10  sysBP            4238 non-null   float64
     11  diaBP            4238 non-null   float64
     12  BMI              4219 non-null   float64
     13  heartRate        4237 non-null   float64
     14  glucose          3850 non-null   float64
     15  TenYearCHD       4238 non-null   int64  
    dtypes: float64(9), int64(7)
    memory usage: 529.9 KB
    


```python
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>male</th>
      <th>age</th>
      <th>education</th>
      <th>currentSmoker</th>
      <th>cigsPerDay</th>
      <th>BPMeds</th>
      <th>prevalentStroke</th>
      <th>prevalentHyp</th>
      <th>diabetes</th>
      <th>totChol</th>
      <th>sysBP</th>
      <th>diaBP</th>
      <th>BMI</th>
      <th>heartRate</th>
      <th>glucose</th>
      <th>TenYearCHD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>39</td>
      <td>4.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>195.0</td>
      <td>106.0</td>
      <td>70.0</td>
      <td>26.97</td>
      <td>80.0</td>
      <td>77.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>46</td>
      <td>2.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>250.0</td>
      <td>121.0</td>
      <td>81.0</td>
      <td>28.73</td>
      <td>95.0</td>
      <td>76.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>48</td>
      <td>1.0</td>
      <td>1</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>245.0</td>
      <td>127.5</td>
      <td>80.0</td>
      <td>25.34</td>
      <td>75.0</td>
      <td>70.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>61</td>
      <td>3.0</td>
      <td>1</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>225.0</td>
      <td>150.0</td>
      <td>95.0</td>
      <td>28.58</td>
      <td>65.0</td>
      <td>103.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>46</td>
      <td>3.0</td>
      <td>1</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>285.0</td>
      <td>130.0</td>
      <td>84.0</td>
      <td>23.10</td>
      <td>85.0</td>
      <td>85.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Checking duplicates
df.duplicated().sum()
```




    0




```python
# Checking missing values
colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
sn.heatmap(df.isnull(), yticklabels=False,cbar=False,cmap=sn.color_palette(colours));
```


![png](output_6_0.png)



```python
# Checking missing values count
df.isnull().sum()
```




    male                 0
    age                  0
    education          105
    currentSmoker        0
    cigsPerDay          29
    BPMeds              53
    prevalentStroke      0
    prevalentHyp         0
    diabetes             0
    totChol             50
    sysBP                0
    diaBP                0
    BMI                 19
    heartRate            1
    glucose            388
    TenYearCHD           0
    dtype: int64




```python
# Percentage missing
count=0
for i in df.isnull().sum(axis=1):
    if i>0:
        count=count+1
print('Total number of rows with missing values is', count)
print('It is',round((count/len(df))*100), 'percent of the entire dataset.')
```

    Total number of rows with missing values is 582
    It is 14 percent of the entire dataset.
    


```python
# Checking distributions of variables with missing values
fig = plt.figure(figsize = (15,20))
ax = fig.gca()
df[['education','cigsPerDay','BPMeds','totChol','BMI','heartRate','glucose']].hist(ax = ax)
```

    <ipython-input-10-de1c6e4a6c98>:4: UserWarning: To output multiple subplots, the figure containing the passed axes is being cleared
      df[['education','cigsPerDay','BPMeds','totChol','BMI','heartRate','glucose']].hist(ax = ax)
    




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x000002A3E8FF3E50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002A3EF9C3F70>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002A3EF9FB400>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x000002A3EFA1A8E0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002A3EFA30100>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002A3EEE54550>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x000002A3EEE540D0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002A3EF5A9880>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000002A3EF5CBD90>]],
          dtype=object)




![png](output_9_2.png)



```python
# Target Variable 
sn.countplot(x='TenYearCHD',data=df,palette='viridis');
```


![png](output_10_0.png)



```python
# Correlation Matrix
plt.figure(figsize=(14,10))
sn.heatmap(df.corr(),cmap= 'Purples',annot =True, linecolor='Green', linewidths=1.0)
plt.title("Correlation among all the Variables before data cleaning", size=20)
plt.show()
```


![png](output_11_0.png)



```python
# Data Imputation
#Columns having mising values are: 'education'(2.5%), 'cigsPerDay'(0.7%), 'BPMeds'(1.3%), 'totChol'(1.2%) and 'glucose'(9.2%). Except the feature glucose all other missing values are less than 2% of data.
#We can drop all other missing values. As for feature glucose, notice in heatmap that glucose is highly correlated with diabetes. 
# Since the count of people at risk is much lesser compared to the count of people at risk and the distribution of glucose is skewed, I will use feature diabetes to fill missing values in glucose.

df.groupby('diabetes').mean()['glucose']
```




    diabetes
    0     79.489186
    1    170.333333
    Name: glucose, dtype: float64




```python
def impute_glucose(cols):
    dia=cols[0]
    glu=cols[1]
    if pd.isnull(glu):
        if dia == 0:
            return 79
        else:
            return 170
    else:
        return glu

df['glucose'] = df[['diabetes','glucose']].apply(impute_glucose,axis=1)
```


```python
#calculate point-biserial correlation
stats.pointbiserialr(df['diabetes'], df['glucose'])
```




    PointbiserialrResult(correlation=0.6251900227640763, pvalue=0.0)




```python
# missing values after glucose imputation
count=0
for i in df.isnull().sum(axis=1):
    if i>0:
        count=count+1
print('Total number of rows with missing values is', count)
print('It is',round((count/len(df))*100), 'percent of the entire dataset.')
```

    Total number of rows with missing values is 251
    It is 6 percent of the entire dataset.
    


```python
# Dropping other missing values
df.dropna(axis = 0, inplace = True) 
print(df.shape)
```

    (3987, 16)
    


```python
# Checking missing values count after imputation
df.isnull().sum()
```




    male               0
    age                0
    education          0
    currentSmoker      0
    cigsPerDay         0
    BPMeds             0
    prevalentStroke    0
    prevalentHyp       0
    diabetes           0
    totChol            0
    sysBP              0
    diaBP              0
    BMI                0
    heartRate          0
    glucose            0
    TenYearCHD         0
    dtype: int64




```python
# Checking imputed dataset
df.shape
```




    (3987, 16)




```python
# Checking duplicates in Imputed dataset
df.duplicated().sum()
```




    0




```python
df.to_csv(r'C:/Users/Lakviru Perera/Desktop/4th YEAR/IS 4007 Statistics in Practice II/Frahmingham/cleaned_df.csv')
```


```python
# Descriptives
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>male</th>
      <th>age</th>
      <th>education</th>
      <th>currentSmoker</th>
      <th>cigsPerDay</th>
      <th>BPMeds</th>
      <th>prevalentStroke</th>
      <th>prevalentHyp</th>
      <th>diabetes</th>
      <th>totChol</th>
      <th>sysBP</th>
      <th>diaBP</th>
      <th>BMI</th>
      <th>heartRate</th>
      <th>glucose</th>
      <th>TenYearCHD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3987.000000</td>
      <td>3987.000000</td>
      <td>3987.000000</td>
      <td>3987.000000</td>
      <td>3987.000000</td>
      <td>3987.000000</td>
      <td>3987.000000</td>
      <td>3987.000000</td>
      <td>3987.000000</td>
      <td>3987.000000</td>
      <td>3987.000000</td>
      <td>3987.000000</td>
      <td>3987.000000</td>
      <td>3987.000000</td>
      <td>3987.00000</td>
      <td>3987.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.433158</td>
      <td>49.478806</td>
      <td>1.981941</td>
      <td>0.491096</td>
      <td>9.020316</td>
      <td>0.029345</td>
      <td>0.005518</td>
      <td>0.309506</td>
      <td>0.025332</td>
      <td>236.620517</td>
      <td>132.222724</td>
      <td>82.861174</td>
      <td>25.774650</td>
      <td>75.873840</td>
      <td>81.66466</td>
      <td>0.149235</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.495574</td>
      <td>8.531588</td>
      <td>1.020696</td>
      <td>0.499983</td>
      <td>11.914558</td>
      <td>0.168794</td>
      <td>0.074087</td>
      <td>0.462348</td>
      <td>0.157152</td>
      <td>44.019766</td>
      <td>21.949243</td>
      <td>11.882166</td>
      <td>4.079846</td>
      <td>12.087463</td>
      <td>22.99468</td>
      <td>0.356365</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>32.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>113.000000</td>
      <td>83.500000</td>
      <td>48.000000</td>
      <td>15.540000</td>
      <td>44.000000</td>
      <td>40.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>42.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>206.000000</td>
      <td>117.000000</td>
      <td>75.000000</td>
      <td>23.060000</td>
      <td>68.000000</td>
      <td>72.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>49.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>234.000000</td>
      <td>128.000000</td>
      <td>82.000000</td>
      <td>25.380000</td>
      <td>75.000000</td>
      <td>79.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>56.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>263.000000</td>
      <td>143.500000</td>
      <td>89.500000</td>
      <td>27.990000</td>
      <td>83.000000</td>
      <td>85.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>70.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>70.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>600.000000</td>
      <td>295.000000</td>
      <td>142.500000</td>
      <td>56.800000</td>
      <td>143.000000</td>
      <td>394.00000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Descriptives for continuous variables 
df.iloc[:,[1,4,9,10,11,12,13,14]].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>cigsPerDay</th>
      <th>totChol</th>
      <th>sysBP</th>
      <th>diaBP</th>
      <th>BMI</th>
      <th>heartRate</th>
      <th>glucose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3987.000000</td>
      <td>3987.000000</td>
      <td>3987.000000</td>
      <td>3987.000000</td>
      <td>3987.000000</td>
      <td>3987.000000</td>
      <td>3987.000000</td>
      <td>3987.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>49.478806</td>
      <td>9.020316</td>
      <td>236.620517</td>
      <td>132.222724</td>
      <td>82.861174</td>
      <td>25.774650</td>
      <td>75.873840</td>
      <td>81.66466</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.531588</td>
      <td>11.914558</td>
      <td>44.019766</td>
      <td>21.949243</td>
      <td>11.882166</td>
      <td>4.079846</td>
      <td>12.087463</td>
      <td>22.99468</td>
    </tr>
    <tr>
      <th>min</th>
      <td>32.000000</td>
      <td>0.000000</td>
      <td>113.000000</td>
      <td>83.500000</td>
      <td>48.000000</td>
      <td>15.540000</td>
      <td>44.000000</td>
      <td>40.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>42.000000</td>
      <td>0.000000</td>
      <td>206.000000</td>
      <td>117.000000</td>
      <td>75.000000</td>
      <td>23.060000</td>
      <td>68.000000</td>
      <td>72.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>49.000000</td>
      <td>0.000000</td>
      <td>234.000000</td>
      <td>128.000000</td>
      <td>82.000000</td>
      <td>25.380000</td>
      <td>75.000000</td>
      <td>79.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>56.000000</td>
      <td>20.000000</td>
      <td>263.000000</td>
      <td>143.500000</td>
      <td>89.500000</td>
      <td>27.990000</td>
      <td>83.000000</td>
      <td>85.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>600.000000</td>
      <td>295.000000</td>
      <td>142.500000</td>
      <td>56.800000</td>
      <td>143.000000</td>
      <td>394.00000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[:,[1,4,9,10,11,12,13,14]].mode().iloc[0,:]
```




    age            40.00
    cigsPerDay      0.00
    totChol       240.00
    sysBP         120.00
    diaBP          80.00
    BMI            22.54
    heartRate      75.00
    glucose        79.00
    Name: 0, dtype: float64



## Univariate Analysis

## Categorical Features


```python
# male
plt.figure(figsize=(8,6), facecolor='w')
sn.countplot(x = 'male', data = df, palette = "rocket",saturation=0.9)
plt.xlabel("male",size=12)
plt.ylabel("count",size=12)
plt.show()
```


![png](output_26_0.png)



```python
# education
plt.figure(figsize=(8,6), facecolor='w')
sn.countplot(x = 'education', data = df, palette = "mako",saturation=0.8)
plt.xlabel("education",size=12)
plt.ylabel("count",size=12)
plt.show()
```


![png](output_27_0.png)



```python
# currentSmoker
plt.figure(figsize=(8,6), facecolor='w')
sn.countplot(x = 'currentSmoker', data = df, palette = "magma",saturation=0.9)
plt.xlabel("currentSmoker",size=12)
plt.ylabel("count",size=12)
plt.show()
```


![png](output_28_0.png)



```python
# BPMeds
plt.figure(figsize=(8,6), facecolor='w')
sn.countplot(x = 'BPMeds', data = df, palette='viridis',saturation=0.9)
plt.xlabel("BPMeds",size=12)
plt.ylabel("count",size=12)
plt.show()
```


![png](output_29_0.png)



```python
# prevalentStroke
plt.figure(figsize=(8,6), facecolor='w')
plt.pie(df["prevalentStroke"].value_counts(),autopct="%1.1f%%",explode=[0,0.01],labels=[0,1],colors=["chocolate","turquoise"])
plt.xlabel("prevalentStroke",size=12)
plt.show()
```


![png](output_30_0.png)



```python
# prevalentHyp
plt.figure(figsize=(8,5), facecolor='w')
sn.countplot(x = 'prevalentHyp', data = df, palette = "icefire",saturation=0.9)
plt.xlabel("prevalentHyp",size=12)
plt.ylabel("count",size=12)
plt.show()
```


![png](output_31_0.png)



```python
# diabetes
plt.figure(figsize=(8,5), facecolor='w')
plt.pie(df["diabetes"].value_counts(),explode=[0,0.01],autopct="%1.1f%%",labels=[0,1],colors=["magenta","gold"])
plt.xlabel("diabetes",size=12)
plt.show()
```


![png](output_32_0.png)


## Numerical/Continuouse Features


```python
# cigsPerDay
plt.figure(figsize=(8, 5), facecolor='w')
plt.subplots_adjust(right=1.5)
plt.subplot(121)
sn.distplot(df['cigsPerDay'],color='tomato')
plt.subplot(122)
sn.boxplot(df['cigsPerDay'],color='tomato')
plt.show()
```


![png](output_34_0.png)



```python
# age
plt.figure(figsize=(8, 5), facecolor='w')
plt.subplots_adjust(right=1.5)
plt.subplot(121)
sn.distplot(df['age'],color='deeppink')
plt.subplot(122)
sn.boxplot(df['age'],color='deeppink')
plt.show()
```


![png](output_35_0.png)



```python
# BMI
plt.figure(figsize=(8, 5), facecolor='w')
plt.subplots_adjust(right=1.5)
plt.subplot(121)
sn.distplot(df['BMI'],color='dodgerblue')
plt.subplot(122)
sn.boxplot(df['BMI'],color='dodgerblue')
plt.show()
```


![png](output_36_0.png)



```python
# totChol
plt.figure(figsize=(8, 5), facecolor='w')
plt.subplots_adjust(right=1.5)
plt.subplot(121)
sn.distplot(df['totChol'],color='green')
plt.subplot(122)
sn.boxplot(df['totChol'],color='green')
plt.show()
```


![png](output_37_0.png)



```python
# sysBP
plt.figure(figsize=(8, 5), facecolor='w')
plt.subplots_adjust(right=1.5)
plt.subplot(121)
sn.distplot(df['sysBP'],color='mediumpurple')
plt.subplot(122)
sn.boxplot(df['sysBP'],color='mediumpurple')
plt.show()
```


![png](output_38_0.png)



```python
# diaBP
plt.figure(figsize=(8, 5), facecolor='w')
plt.subplots_adjust(right=1.5)
plt.subplot(121)
sn.distplot(df['diaBP'],color='gold')
plt.subplot(122)
sn.boxplot(df['diaBP'],color='gold')
plt.show()
```


![png](output_39_0.png)



```python
# heartRate
plt.figure(figsize=(8, 5), facecolor='w')
plt.subplots_adjust(right=1.5)
plt.subplot(121)
sn.distplot(df['heartRate'],color='crimson')
plt.subplot(122)
sn.boxplot(df['heartRate'],color='crimson')
plt.show()
```


![png](output_40_0.png)



```python
# glucose
plt.figure(figsize=(8, 5), facecolor='w')
plt.subplots_adjust(right=1.5)
plt.subplot(121)
sn.distplot(df['glucose'],color='turquoise')
plt.subplot(122)
sn.boxplot(df['glucose'],color='turquoise')
plt.show()
```


![png](output_41_0.png)



```python
# Target Variable -  TenYearCHD 
plt.figure(figsize=(8,5), facecolor='w')
plt.subplots_adjust(right=1.5)
plt.subplot(121)
sn.countplot(x="TenYearCHD", data=df,palette='viridis')
plt.xlabel("TenYearCHD ",size=12)
plt.ylabel("count",size=12)
plt.subplot(122)
labels=[0,1]
plt.pie(df["TenYearCHD"].value_counts(),explode=[0,0.01],autopct="%1.1f%%",labels=labels,colors=["slateblue","mediumseagreen"])
plt.show()
```


![png](output_42_0.png)



```python
df["TenYearCHD"].value_counts()
```




    0    3392
    1     595
    Name: TenYearCHD, dtype: int64



## Bivariate Analysis


```python
# Numeric data
df_numeric = df.iloc[:,[1,4,9,10,11,12,13,14,15]]
```


```python
# Heatmap for continuous data
plt.figure(figsize=(12,8), facecolor='w')
sn.heatmap(df_numeric.corr(),cmap='rocket_r',annot=True,linecolor='Green', linewidths=1.0)
plt.title("Correlations for continuous Variables", size=16)
```




    Text(0.5, 1.0, 'Correlations for continuous Variables')




![png](output_46_1.png)



```python
# Categorical Data
df_cat = df.iloc[:,[0,2,3,5,6,7,8,15]]
```


```python
from scipy.stats import chi2_contingency

factors_paired = [(i,j) for i in df_cat.columns.values for j in df_cat.columns.values] 

chi2, p_values =[], []

for f in factors_paired:
    if f[0] != f[1]:
        chitest = chi2_contingency(pd.crosstab(df_cat[f[0]], df_cat[f[1]]))   
        chi2.append(chitest[0])
        p_values.append(chitest[1])
    else:      # for same factor pair
        chi2.append(0)
        p_values.append(0)

chi2 = np.array(chi2).reshape((8,8)) # shape it as a matrix
chi2 = pd.DataFrame(chi2, index=df_cat.columns.values, columns=df_cat.columns.values) # then a df for convenience
sn.heatmap(chi2.corr(),cmap='Purples',annot=True, linecolor='Blue', linewidths=1.0)
plt.title("Correlations for categorical Variables", size=16)
```




    Text(0.5, 1.0, 'Correlations for categorical Variables')




![png](output_48_1.png)



```python
# Heatmap for categorical data
plt.figure(figsize=(12,8), facecolor='w')
sn.heatmap(df_cat.corr(),cmap='Purples',annot=True, linecolor='Blue', linewidths=1.0)
plt.title("Correlations for categorical Variables", size=16)
```




    Text(0.5, 1.0, 'Correlations for categorical Variables')




![png](output_49_1.png)



```python
# Heatmap for categorical data
plt.figure(figsize=(12,8), facecolor='w')
sn.heatmap(df_cat.corr(),cmap='Purples',annot=True, linecolor='Blue', linewidths=1.0)
plt.title("Correlations for categorical Variables", size=16)
```




    Text(0.5, 1.0, 'Correlations for categorical Variables')




![png](output_50_1.png)



```python
sn.pairplot(df,palette='set3')
plt.show()
```


![png](output_51_0.png)



```python
# male vs TenYearCHD
plt.figure(figsize=(8,5), facecolor='w')
sn.catplot(data=df,x='male',hue='TenYearCHD',kind='count',palette='cubehelix',height=5)
plt.ylabel('No. of heart patients')
plt.xlabel("male\n0 is female and 1 is male",size=12)
plt.show();
```


    <Figure size 576x360 with 0 Axes>



![png](output_52_1.png)



```python
# age vs TenYearCHD
plt.figure(figsize=(8,5), facecolor='w')
sn.boxplot(df['TenYearCHD'],df['age'])
plt.show();
```


![png](output_53_0.png)



```python
# education vs TenYearCHD
plt.figure(figsize=(8,5))
sn.catplot(data=df,x='TenYearCHD',hue='education',kind='count',palette='seismic',height=5)
plt.show();
```


    <Figure size 576x360 with 0 Axes>



![png](output_54_1.png)



```python
# currentSmoker vs TenYearCHD
plt.figure(figsize=(8,5), facecolor='w')
sn.catplot(data=df,x='TenYearCHD',hue='currentSmoker',kind='count',palette='Dark2_r',height=5)
plt.show();
```


    <Figure size 576x360 with 0 Axes>



![png](output_55_1.png)



```python
# cigsPerDay vs TenYearCHD
plt.figure(figsize=(8,5), facecolor='w')
plt.figure(figsize=(8,5), facecolor='w')
sn.violinplot(x="TenYearCHD", y="cigsPerDay", data=df,palette='rainbow')
plt.show();
```


    <Figure size 576x360 with 0 Axes>



![png](output_56_1.png)



```python
# BPMeds vs TenYearCHD
bpChd = pd.crosstab(index=df.TenYearCHD,columns=df.BPMeds,normalize=True)
bpChdPercent = 100*bpChd
bpChdPercent.plot(kind='barh',stacked=True,figsize=(12,5),colormap='Spectral_r')
plt.xlabel('Percent')
plt.show();
```


![png](output_57_0.png)



```python
# prevalentStroke vs TenYearCHD
strokeChd = pd.crosstab(index=df.TenYearCHD,columns=df.prevalentStroke,normalize=True) 
strokeChdPercent = 100*strokeChd
strokeChdPercent
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>prevalentStroke</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>TenYearCHD</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>84.725357</td>
      <td>0.351141</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14.722849</td>
      <td>0.200652</td>
    </tr>
  </tbody>
</table>
</div>




```python
strokeChdPercent.plot(kind='barh',stacked=True,figsize=(20,5),colormap='winter_r',edgecolor="0.8")
plt.xlabel('Percent')
plt.show();
```


![png](output_59_0.png)



```python
# prevalentHyp vs TenYearCHD
plt.figure(figsize=(8,5), facecolor='w')
sn.catplot(data=df,x='TenYearCHD',hue='prevalentHyp',kind='count',palette='gnuplot2',height=5,saturation=0.9,edgecolor="0.6")
plt.show();
```


    <Figure size 576x360 with 0 Axes>



![png](output_60_1.png)



```python
# diabetes vs TenYearCHD
plt.figure(figsize=(8,5), facecolor='w')
sn.catplot(data=df,y='TenYearCHD',hue='diabetes',kind='count',palette='hsv',height=5,saturation=0.4)
plt.show();
```


    <Figure size 576x360 with 0 Axes>



![png](output_61_1.png)



```python
# totChol vs TenYearCHD 
plt.figure(figsize=(8,5), facecolor='w')
sn.violinplot(x="TenYearCHD", y="totChol", data=df,palette='Paired_r')
sn.catplot(x='TenYearCHD',y='totChol',palette='gnuplot_r',data=df)
plt.show()
```


![png](output_62_0.png)



![png](output_62_1.png)



```python
# sysBP vs TenYearCHD  
plt.figure(figsize=(8,5), facecolor='w')
sn.boxplot(x='TenYearCHD',y='sysBP',data=df,palette='gist_rainbow_r')
plt.show();
```


![png](output_63_0.png)



```python
# diaBP vs TenYearCHD  
plt.figure(figsize=(8,5), facecolor='w')
bp0=df[df['TenYearCHD']==0]['diaBP']
bp1=df[df['TenYearCHD']==1]['diaBP']
plt.figure(figsize=(8,5), facecolor='w')
plt.hist(bp0, bins=100, alpha=0.5,color='limegreen',histtype='barstacked')
plt.hist(bp1, bins=100, alpha=0.5,color='blue',histtype='barstacked')
plt.legend(labels=['Patients without risk','Patients with risk'])
plt.xlabel('diaBP')
plt.show();
```


    <Figure size 576x360 with 0 Axes>



![png](output_64_1.png)



```python
# BMI vs TenYearCHD 
plt.figure(figsize=(8,5), facecolor='w')
sn.violinplot(x="TenYearCHD", y="BMI", data=df,palette='rocket')
plt.show();
```


![png](output_65_0.png)



```python
# heartRate vs TenYearCHD 
plt.figure(figsize=(8,5), facecolor='w')
rate0=df[df['TenYearCHD']==0]['heartRate']
rate1=df[df['TenYearCHD']==1]['heartRate']
plt.figure(figsize=(8,5), facecolor='w')
plt.hist(rate0, bins=100, alpha=0.5,color='coral',histtype='barstacked')
plt.hist(rate1, bins=100, alpha=0.5,color='navy',histtype='barstacked')
plt.legend(labels=['Patients without risk','Patients with risk'])
plt.xlabel('heartRate')
plt.show();
```


    <Figure size 576x360 with 0 Axes>



![png](output_66_1.png)



```python
# glucose vs TenYearCHD  
plt.figure(figsize=(8,5), facecolor='w')
sn.boxplot(x='TenYearCHD',y='glucose',data=df,palette='CMRmap_r')
plt.show();
```


![png](output_67_0.png)



```python
# age vs totoChol
def classify_age(age):
    if(age<35):
        return ("Young")
    elif((age>=35) & (age<50)):
        return("Middle age(40-50)")
    elif((age>=50)& (age<60)):
        return("Post 50(50-60)")
    else:
        return("Old(>60)")
    
age_class= df['age'].apply(classify_age)

plt.figure(figsize=(10,6), facecolor='w')
sn.boxplot(x=age_class,y="totChol",order=['Young','Middle age(40-50)','Post 50(50-60)','Old(>60)'],data=df)
plt.title("Distribution of age with respect to totChol", size=20)
plt.xlabel('age_class',size=14)
plt.ylabel('totChol',size=14)
plt.show();
```


![png](output_68_0.png)



```python
plt.scatter('age','totChol',data=df)
```




    <matplotlib.collections.PathCollection at 0x2a3920ae0a0>




![png](output_69_1.png)


## Multivariate Analysis


```python
plt.figure(figsize=(8,5), facecolor='w')
sn.boxplot(x="prevalentHyp", y="heartRate",hue="TenYearCHD",data=df,palette="autumn",saturation=0.8)
plt.show();
```


![png](output_71_0.png)



```python
# boxplots for the relationship among age,sysBP and TenYearCHD
# Function to classify people based on their age.
def classify_age(age):
    if(age<35):
        return ("Young")
    elif((age>=35) & (age<50)):
        return("Middle age(40-50)")
    elif((age>=50)& (age<60)):
        return("Post 50(50-60)")
    else:
        return("Old(>60)")
    
age_class= df['age'].apply(classify_age)
    
plt.figure(figsize=(10,5))
sn.boxplot(age_class,'sysBP', hue="TenYearCHD", order=['Young','Middle age(40-50)','Post 50(50-60)','Old(>60)'],data=df,palette='cool_r',saturation=0.8)
plt.show();
```


![png](output_72_0.png)



```python
# linegraph to check the relationship between age and cigsPerDay, totChol, glucose
cigs = df.groupby("age").cigsPerDay.mean()
chol = df.groupby("age").totChol.mean()
glu  = df.groupby("age").glucose.mean()

plt.figure(figsize=(10,6), facecolor='w')
sn.lineplot(data=cigs, label="cigsPerDay")
sn.lineplot(data=chol, label="totChol")
sn.lineplot(data=glu, label="glucose")
plt.title("Graph showing cigsPerDay,totChol and glucose in every age group", size=16)
plt.xlabel("age", size=15)
plt.ylabel("count", size=15)
plt.xticks(size=12)
plt.yticks(size=12);
```


![png](output_73_0.png)



```python
#sysBP vs diaBP with respect to currentSmoker and male attributes
plt.figure(figsize=(12, 8), facecolor='w')
sn.lmplot('sysBP', 'diaBP', data=df, hue="TenYearCHD", col="male",row="currentSmoker",palette='gist_earth')
plt.show();
```


    <Figure size 864x576 with 0 Axes>



![png](output_74_1.png)


## Advanced Analysis


```python
# Outlier Detection
plt.figure(figsize=(20,8), facecolor='w')
sn.boxplot(data=df_numeric)
plt.show()
```


![png](output_76_0.png)



```python
# making a copy of the dataset
df_copy = df.copy()
df_copy.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>male</th>
      <th>age</th>
      <th>education</th>
      <th>currentSmoker</th>
      <th>cigsPerDay</th>
      <th>BPMeds</th>
      <th>prevalentStroke</th>
      <th>prevalentHyp</th>
      <th>diabetes</th>
      <th>totChol</th>
      <th>sysBP</th>
      <th>diaBP</th>
      <th>BMI</th>
      <th>heartRate</th>
      <th>glucose</th>
      <th>TenYearCHD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>39</td>
      <td>4.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>195.0</td>
      <td>106.0</td>
      <td>70.0</td>
      <td>26.97</td>
      <td>80.0</td>
      <td>77.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>46</td>
      <td>2.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>250.0</td>
      <td>121.0</td>
      <td>81.0</td>
      <td>28.73</td>
      <td>95.0</td>
      <td>76.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>48</td>
      <td>1.0</td>
      <td>1</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>245.0</td>
      <td>127.5</td>
      <td>80.0</td>
      <td>25.34</td>
      <td>75.0</td>
      <td>70.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>61</td>
      <td>3.0</td>
      <td>1</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>225.0</td>
      <td>150.0</td>
      <td>95.0</td>
      <td>28.58</td>
      <td>65.0</td>
      <td>103.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>46</td>
      <td>3.0</td>
      <td>1</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>285.0</td>
      <td>130.0</td>
      <td>84.0</td>
      <td>23.10</td>
      <td>85.0</td>
      <td>85.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Outlier detection in totChol 
df_copy['totChol'].max()
```




    600.0




```python
# Outlier detection in sysBP
df_copy['sysBP'].max()
```




    295.0




```python
# Removing outliers
df_copy = df_copy[df_copy['totChol']<600.0]
df_copy = df_copy[df_copy['sysBP']<295.0]
df_copy.shape
```




    (3985, 16)



### Resampling imbalanced dataset by oversampling positive cases


```python
target1=df_copy[df_copy['TenYearCHD']==1]
target0=df_copy[df_copy['TenYearCHD']==0]
```


```python
target1=resample(target1,replace=True,n_samples=len(target0),random_state=40)
```


```python
target=pd.concat([target0,target1])
```


```python
target['TenYearCHD'].value_counts()
```




    1    3392
    0    3392
    Name: TenYearCHD, dtype: int64




```python
balanced_df=target
balanced_df.shape
```




    (6784, 16)




```python
#Distribution of heart disease cases in the balanced dataset, the outcome variable
plt.figure(figsize=(8,5), facecolor='w')
plt.subplots_adjust(right=1.5)
plt.subplot(121)
sn.countplot(x="TenYearCHD", data=balanced_df,palette='winter')
plt.title("Count of TenYearCHD column", size=14)
plt.subplot(122)
labels=[0,1]
plt.pie(balanced_df["TenYearCHD"].value_counts(),autopct="%1.1f%%",labels=labels,colors=["blue","lime"])
plt.show()
```


![png](output_87_0.png)



```python
#To idenfify the features that have larger contribution towards the outcome variable, TenYearCHD
X=balanced_df.iloc[:,0:15]
y=balanced_df.iloc[:,-1]
print("X - ", X.shape, "\ny - ", y.shape)
```

    X -  (6784, 15) 
    y -  (6784,)
    

### Feature Selection


```python
#Apply SelectKBest and extract top 10 features
best=SelectKBest(score_func=chi2, k=10)
```


```python
fit=best.fit(X,y)
```


```python
data_scores=pd.DataFrame(fit.scores_)
data_columns=pd.DataFrame(X.columns)
```


```python
#Join the two dataframes
scores=pd.concat([data_columns,data_scores],axis=1)
scores.columns=['Feature','Score']
print(scores.nlargest(11,'Score'))
```

             Feature        Score
    10         sysBP  2127.940554
    14       glucose  1172.248804
    1            age  1004.535192
    4     cigsPerDay   790.532235
    9        totChol   766.291552
    11         diaBP   488.928591
    7   prevalentHyp   221.768997
    0           male    66.816351
    5         BPMeds    66.216216
    8       diabetes    54.258065
    12           BMI    44.855037
    


```python
#To visualize feature selection
scores = scores.sort_values(by="Score", ascending=False)
plt.figure(figsize=(20,7), facecolor='w')
sn.barplot(x='Feature',y='Score',data=scores,palette='Oranges_r')
plt.title("Plot showing the best features in descending order", size=20)
plt.show()
```


![png](output_94_0.png)



```python
#Select 10 features
features=scores["Feature"].tolist()[:10]
features
```


```python
featured_df=balanced_df[['sysBP','glucose','age','cigsPerDay','totChol','diaBP','prevalentHyp','male','BPMeds','diabetes','TenYearCHD']]
featured_df.head()
```


```python
# VIF for selected features
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["feature"] = featured_df.drop(columns=['TenYearCHD']).columns
vif_data["VIF"] = [variance_inflation_factor(featured_df.drop(columns=['TenYearCHD']).values, i)
                          for i in range(len(featured_df.drop(columns=['TenYearCHD']).columns))]
vif_data
```

### Model fitting usins statsmodels

#### Splitting data


```python
train = featured_df.sample(frac=0.8,random_state=1)
test = featured_df.drop(train.index)
```

#### model 1


```python
x1=train[['sysBP','glucose','age','cigsPerDay','totChol','diaBP','prevalentHyp','male','BPMeds','diabetes']]
y1 = train['TenYearCHD']
```


```python
# Logistic regresseion using statsmodels   
x1_constant= sm.add_constant(x1)
log_reg = sm.Logit(y1,x1_constant).fit() 
log_reg.summary()
```

#### model 2


```python
x2 = train[['sysBP','glucose','age','cigsPerDay','totChol','prevalentHyp','male','BPMeds','diabetes']]
y2 = train['TenYearCHD']
```


```python
# Logistic regresseion using statsmodels  
x2_constant= sm.add_constant(x2)
log_reg = sm.Logit(y2,x2_constant).fit() 
log_reg.summary()
```

#### model3


```python
x3 = train[['sysBP','glucose','age','cigsPerDay','totChol','prevalentHyp','male','BPMeds']]
y3 = train['TenYearCHD']
```


```python
# Logistic regresseion using statsmodels 
x3_constant= sm.add_constant(x3)
log_reg = sm.Logit(y3,x3_constant).fit() 
log_reg.summary()
```


```python
#### model 4
```


```python
x4 = train[['sysBP','glucose','age','cigsPerDay','totChol','prevalentHyp','male']]
y4 = train['TenYearCHD']
```


```python
# Vif for selected features in the final model
vif_data2 = pd.DataFrame()
vif_data2["feature"] = x4.columns
vif_data2["VIF"] = [variance_inflation_factor(x4.values, i)
                          for i in range(len(x4.columns))]
vif_data2
```


```python
# Logistic regresseion using statsmodels   
x4_constant= sm.add_constant(x4)
log_reg = sm.Logit(y4,x4_constant).fit() 
log_reg.summary()
```

### Model fitting using sklearn

#### Splitting


```python
y = featured_df['TenYearCHD']
X = featured_df[['sysBP','glucose','age','cigsPerDay','totChol','prevalentHyp','male']]
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)
```

#### Scaling


```python
scaler = MinMaxScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)
```

#### Model Fitting


```python
lr = LogisticRegression(random_state=1)
model = lr.fit(train_x, train_y)
lr_predict = lr.predict(test_x)
```


```python
# Evaluating the model
lr_conf_matrix = confusion_matrix(test_y, lr_predict)
lr_acc_score = accuracy_score(test_y, lr_predict)
print("confusion matrix")
print(lr_conf_matrix)
print("\n")
print("Accuracy of Logistic Regression:",lr_acc_score*100,'\n')
print(classification_report(test_y,lr_predict))
```


```python
# heatmap of confusion matrix
log_conf_matrix = pd.DataFrame(data =lr_conf_matrix,  
                           columns = ['Predicted:0', 'Predicted:1'],  
                           index =['Actual:0', 'Actual:1']) 
plt.figure(figsize = (8,6)) 
sn.heatmap(log_conf_matrix,annot=True,cmap='RdBu_r')
plt.show();  
```


```python
TN=lr_conf_matrix[0,0]
TP=lr_conf_matrix[1,1]
FN=lr_conf_matrix[1,0]
FP=lr_conf_matrix[0,1]
sensitivity=round(TP/float(TP+FN),4)
print('Model sensitivity : ',sensitivity)
specificity=round(TN/float(TN+FP),4)
print('Model specificity : ',specificity)
```


```python
# ROC curve
fpr, tpr, thresholds = roc_curve(test_y,lr_predict)
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Heart disease classifier')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)
```


```python
import sklearn
sklearn.metrics.roc_auc_score(test_y,lr_predict)
```


```python
# model coefficients
coeffecients = pd.DataFrame(lr.coef_.ravel(),X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients.sort_values(by=['Coeffecient'],inplace=True,ascending=False)
coeffecients
```


```python
# model intercept
print('Intercept :',lr.intercept_)
```

### KNeighbors Classifier


```python
knn = KNeighborsClassifier(n_neighbors=1)
model = knn.fit(train_x, train_y)
knn_predict = knn.predict(test_x)
knn_conf_matrix = confusion_matrix(test_y, knn_predict)
knn_acc_score = accuracy_score(test_y, knn_predict)
print("confussion matrix")
print(knn_conf_matrix)
print("\n")
print("Accuracy of k-NN Classification:",knn_acc_score*100,'\n')
print(classification_report(test_y, knn_predict))
```


```python
Knn_conf = pd.DataFrame(data =knn_conf_matrix,  
                           columns = ['Predicted:0', 'Predicted:1'],  
                           index =['Actual:0', 'Actual:1']) 
plt.figure(figsize = (8,6)) 
sn.heatmap(Knn_conf,annot=True,cmap='inferno_r')
plt.show();  
```


```python
TN=knn_conf_matrix[0,0]
TP=knn_conf_matrix[1,1]
FN=knn_conf_matrix[1,0]
FP=knn_conf_matrix[0,1]
sensitivity=round(TP/float(TP+FN),4)
print('Model sensitivity : ',sensitivity)
specificity=round(TN/float(TN+FP),4)
print('Model specificity : ',specificity)
```
