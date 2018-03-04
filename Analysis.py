import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#df_swing = pd.read_csv('2008_swing_states.csv')

#df_swing[['state', 'county', 'dem_share']]

#sns.set()
#a = plt.hist(df_swing['dem_share'])
#a = plt.xlabel('percent of vote for Obama')
#a = plt.ylabel('number of counties')

#plt.show()

''' 
In this program we will:
    * import test data as data frame
    * construct Wilcoxon Sign Test function
    * apply bootstrap on test data with Wilcoxon Sign Test
'''

'''
Function to calculate Wilcoxon Sign Test from a list of already compared data.
'''
def Wilcoxon_From_ComparedList(WList):    
    WList = np.array(WList)
    WList = WList[WList != 0]
    n = len(WList)
    abslist = np.abs(WList)
    signList = map(lambda x: 1 if x > 0 else -1, list(WList))    
    allList = list(zip(np.array(abslist),np.fromiter(signList, dtype=np.int),  WList))
    allList.sort(key=lambda x: x[0])
    allList = np.array(allList)
    WResult = []
    mPos = 1
    for position, item in enumerate(allList):
        count = list(allList[:, 2]).count(item[2]) 
        if (count == 1):
            WResult.append((position+1) * item[1])
            mPos = 1
        elif (count > 1 & mPos == 1):
            totalCo = np.array(range(count)) + 1
            WResult.append((count*position+totalCo.sum()) * item[1] / count)
            mPos = mPos + 1
        else:
            WResult.append(WResult[len(WResult) - 1])
    d = np.sqrt(n*(n+1)*(2*n+1)/6)
    w = sum(WResult)
    z = (w-0.5)/d
    print('w:{0}, d:{1}, z{2}'.format(w,d,z))
    return z

'''
Function to perform bootstrap of selected data.
'''
def bootstrap_replicate_ld(data, func):
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)

if __name__ == '__main__':
    wilcoxonData = pd.read_csv('WilcoxonData.csv')

    print(wilcoxonData['HT - % diff Car'])
    Wilcoxon_From_ComparedList(wilcoxonData['HT - % diff Car'])

    BS_Size = 10000
    HT_C_BS = np.empty(BS_Size)
    HT_L_BS = np.empty(BS_Size)

    LT_C_BS = np.empty(BS_Size)
    LT_H_BS = np.empty(BS_Size)

    C_LT_BS = np.empty(BS_Size)
    C_HT_BS = np.empty(BS_Size)

    for i in range(BS_Size):
        HT_C_BS[i] = bootstrap_replicate_ld(wilcoxonData['HT - % diff Car'], Wilcoxon_From_ComparedList)
        HT_L_BS[i] = bootstrap_replicate_ld(wilcoxonData['HT - % diff LT'], Wilcoxon_From_ComparedList)

        LT_C_BS[i] = bootstrap_replicate_ld(wilcoxonData['LT - % diff Car'], Wilcoxon_From_ComparedList)
        LT_H_BS[i] = bootstrap_replicate_ld(wilcoxonData['LT - % diff HT'], Wilcoxon_From_ComparedList)

        C_LT_BS[i] = bootstrap_replicate_ld(wilcoxonData['Car - % diff LT'], Wilcoxon_From_ComparedList)
        C_HT_BS[i] = bootstrap_replicate_ld(wilcoxonData['Car - % diff HT'], Wilcoxon_From_ComparedList)

    finalData = pd.DataFrame(HT_C_BS, columns=['z'])
    finalData['experiment'] = 'Est result better than Car'
    ltData = pd.DataFrame(HT_L_BS, columns=['z'])
    ltData['experiment'] = 'Est result better than Light Truck'
    finalData = finalData.append(ltData)
    bx = sns.boxplot(x='experiment', y='z', data=finalData)
    bx.set_xlabel('Compare to different vehicles as baseline')
    bx.set_title('Heavy Truck Validations (bootstrap 10000)')
    plt.show()

    finalData = pd.DataFrame(LT_C_BS, columns=['z'])
    finalData['experiment'] = 'Est result better than Car'
    ltData = pd.DataFrame(LT_H_BS, columns=['z'])
    ltData['experiment'] = 'Est result better than Heavy Truck'
    finalData = finalData.append(ltData)
    bx = sns.boxplot(x='experiment', y='z', data=finalData)
    bx.set_xlabel('Estimation better than different vehicle')
    bx.set_title('Light Truck Validations (bootstrap 10000)')
    plt.show()
    
    finalData = pd.DataFrame(C_LT_BS, columns=['z'])
    finalData['experiment'] = 'Est result better than Light Truck'
    ltData = pd.DataFrame(C_HT_BS, columns=['z'])
    ltData['experiment'] = 'Est result better than Heavy Truck'
    finalData = finalData.append(ltData)
    bx = sns.boxplot(x='experiment', y='z', data=finalData)
    bx.set_xlabel('Estimation better than different vehicle')
    bx.set_title('Passenger Car Validations (bootstrap 10000)')
    plt.show()
    
    #sns.boxplot(data=HT_C_BS)
    #sns.boxplot(data=HT_L_BS)

    #f,axes = plt.subplots(1,2)
    #axes[0].set_title('Estimaton better than Passenger Car basline')
    #sns.boxplot(data=LT_C_BS, ax=axes[0])
       
    #sns.boxplot(data=LT_H_BS, ax=axes[1])
    #axes[1].set_title('Estimaton better than Heavy Truck basline')
    #plt.suptitle('Light Truck')

    #sns.boxplot(data=C_LT_BS)
    #sns.boxplot(data=C_HT_BS)

    

    




