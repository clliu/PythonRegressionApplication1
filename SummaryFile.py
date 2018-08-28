import pandas as pd
df = pd.read_csv('literary_birth_rate.csv')
df.head()
df.tail()
df.columns
df.shape
df.info()
df.ColumnName.value_counts(dropna=False)
df['ColumnName'].value_counts(dropna=False)
df.describe()
'''
        Column_Name     population
count   164.000000      1.220000e+02
mean    80.301220       6.345768e+07
std     22.977265       2.605977e+08
min     12.600000       1.035660e+05
25%     66.675000       3.778175e+06
50%     90.200000       9.995450e+06
75%     98.500000       2.642217e+07
max     100.000000      2.313000e+09
'''

'''
pd.melt(frame=df, id_vars='name',
value_vars=['treatment a', 'treatment b’],
var_name='treatment', value_name='result')

weather2_tidy = weather.pivot_table(values='value',
                                    index='date',
                                    columns='element',
                                    aggfunc=np.mean)

import numpy as np
data = np.loadtxt('FileName.txt', delimiter=',')

#optional parameters
data = np.loadtxt('FileName.txt', delimiter=',',
                  skiprows=1,
                  usecols=[0,2,3],
                  dtype=str)
'''
'''Excel file import 
'''
import pandas as pd

file = 'urbanpop.xlsx'
data = pd.ExcelFile(file)
print(data.sheet_names)
['1960-1966', '1967-1974', '1975-2011']
df1 = data.parse('1960-1966') #sheet name, as a string
df2 = data.parse(0)           #sheet index, as a float

'''
SAS file import

import pandas as pd
from sas7bdat import SAS7BDAT
with SAS7BDAT('urbanpop.sas7bdat') as file:
df_sas = file.to_data_frame()
'''
''' Stata file
'''
import pandas as pd
data = pd.read_stata('urbanpop.dta')

'''Teradata'''

'''Teradata module
import teradata
import pandas as pd

host,username,password = 'HOST','UID', 'PWD'
#Make a connection
udaExec = teradata.UdaExec (appName="test", version="1.0", logConsole=False)


with udaExec.connect(method="odbc",system=host, username=username,
                            password=password, driver="DRIVERNAME") as connect:

    query = "SELECT * FROM DATABASEX.TABLENAMEX;"

    #Reading query to df
    df = pd.read_sql(query,connect)

# do something with df,e.g.
print(df.head()) #to see the first 5 rows
'''
'''pyodbc module'''
import pyodbc

 #You can install teradata via PIP: pip install pyodbc
 #to get a list of your odbc drivers names, you could do: pyodbc.drivers()

#Make a connection
link = 'DRIVER=DRIVERNAME;DBCNAME=%s;UID=%s;PWD=%s'%(host, username, password)
with pyodbc.connect(link,autocommit=True) as connect:

    #Reading query to df
    df = pd.read_sql(query,connect)

'''sqlalchemy Module'''
from sqlalchemy import create_engine

#Make a connection
link = 'teradata://'+ username +':'+ password +'@'+host+'/'+'?driver=DRIVERNAME'
with create_engine(link) as connect:

    #Reading query to df
    df = pd.read_sql(query,connect)

''' Merge Data '''
concatenated = pd.concat([weather_p1, weather_p2])
print(concatenated)

'''
date         element     value
0 2010-01-30   tmax      27.8
1 2010-01-30   tmin      14.5
0 2010-02-02   tmax      27.3
1 2010-02-02   tmin      14.4
'''

concatenated = concatenated.loc[0, :]
'''
index    date    element  value
  0   2010-01-30   tmax   27.8
  0   2010-02-02   tmax   27.3
'''

pd.concat([weather_p1, weather_p2], ignore_index=True)
'''
index   date     element  value
  0   2010-01-30   tmax   27.8
  1   2010-01-30   tmin   14.5
  2   2010-02-02   tmax   27.3
  3   2010-02-02   tmin   14.4
'''

import glob
csv_files = glob.glob('*.csv')
list_data = []

for filename in csv_files:
    data = pd.read_csv(filename)
    list_data.append(data)

pd.concat(list_data)

'''Merge data'''
pd.merge(left=state_populations, right=state_codes,
         on=None, left_on='state', right_on='name')
'''
index     state     population_2016   name      ANSI
  0     California    39250017      California   CA
  1     Texas         27862596      Texas        TX
  2     Florida       20612439      Florida      FL
  3     New York      19745289      New York     NY
'''

'''Line plot'''
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 1, 201)
y = np.sin((2*np.pi*x)**2)
plt.plot(x, y, 'red')
plt.show()

'''Scatter plot'''
import numpy as np
import matplotlib.pyplot as plt
x = 10*np.random.rand(200,1)
y = (0.2 + 0.8*x) * np.sin(2*np.pi*x) + np.random.randn(200,1)
plt.scatter(x,y)
plt.show()

'''Histograms'''
import numpy as np
import matplotlib.pyplot as plt
x = 10*np.random.rand(200,1)
y = (0.2 + 0.8*x) * np.sin(2*np.pi*x) + np.random.randn(200,1)
plt.hist(y, bins=20)
plt.show()

'''Common axis'''
import matplotlib.pyplot as plt
plt.plot(t, temperature, 'red')
plt.plot(t, dewpoint, 'blue') # Appears on same axes
plt.xlabel('Date')
plt.title('Temperature & Dew Point')
plt.show() # Renders plot objects to screen