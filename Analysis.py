import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_swing = pd.read_csv('2008_swing_states.csv')

df_swing[['state', 'county', 'dem_share']]

sns.set()
a = plt.hist(df_swing['dem_share'])
a = plt.xlabel('percent of vote for Obama')
a = plt.ylabel('number of counties')

plt.show()
