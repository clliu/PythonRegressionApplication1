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

'''
Function to plot diagram for different properties
'''
def PlotDistributionDiagramForProperty(vehiclePropery, x='WGTCDTR_D', y='ORIGAVTW', xlabel='', ylabel='Original Average Track Width'):
    #vp = sns.boxplot(x=x, y=y, data=vehiclePropery)
    vp = sns.violinplot(x=x, y=y, data=vehiclePropery)
    #vp = sns.violinplot(x='WGTCDTR_D', y='ORIGAVTW', data=vehiclePropery, inner=None,color='lightgray')
    #sp = sns.stripplot(x='WGTCDTR_D', y='ORIGAVTW', data=vehiclePropery, size=4, jitter=True)
    vp.set_title('{0} of Different Vehicle Type'.format(ylabel))
    vp.set_xlabel(xlabel)
    vp.set_ylabel(ylabel)
    plt.show()    


def StatisticFigure(vehiclePropery, x='WGTCDTR_D', y='ORIGAVTW'):
    vehicleTypes = np.unique(vehiclePropery[x])

    print('{0}'.format(y))
    for v in vehicleTypes:
        d = vehiclePropery[y][vehiclePropery[x] == v]
        print('{0}: mean:{1}, std:{2}, median:{3}, min:{4}, max:{5}\r\n'.format(v, np.mean(d), np.std(d), np.median(d), np.min(d), np.max(d)))

def ExtractVehiclePropertyForVehicleType(vehiclePropery, x='WGTCDTR_D', y='ORIGAVTW'):
    vehicleTypes = np.unique(vehiclePropery[x])

    yList = {}
    for v in vehicleTypes:
        yList[v] = vehiclePropery[y][vehiclePropery[x] == v]
    return yList

'''
Remove 0 rows
'''
def ReplaceZeroToNanRows(df, colName):
    df[colName].replace(0, np.nan, inplace=True)    
    return df

'''
Perform variable estimation based on input data and reutrn 3 variables for Passenger Car, Light Truck and Heavy Truck respectively.
    Expecting data frame in following columns:
        Accident Type,MAIS,PassengerCar_Count,PassengerCar_%,LightTruck_Count,LightTruck_%,HeavyTruck_Count,HeavyTruck_%
    Output: a, b, c
        Where a indicate the % for LT for Passenger car estimation, (1-a) incidate teh % for HT for Car
              b indicate the % for Car for LT, and (a-b) for HT on LT
              c indicate the % for Car for HT and (1-c) for LT on HT
'''
def EstimationBasedOnErrorRate(df, car='PassengerCar_%', LT='LightTruck_%', HT='HeavyTruck_%'):
    a = 0.5

'''
Function to plot box diagram for different vehicle validations.
'''
def EstimationValidation(bootscrap_size=10000):
    wilcoxonData = pd.read_csv('WilcoxonData.csv')

    print(wilcoxonData['HT - % diff Car'])
    Wilcoxon_From_ComparedList(wilcoxonData['HT - % diff Car'])

    BS_Size = bootscrap_size
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

'''
Function to estimate later half of relationship with earlier half of data.
'''
def EstimateCoefficient():
    vehiclePropery = pd.read_csv('vehcileProperies.csv')
    #PlotDistributionDiagramForProperty(vehiclePropery,  x='WGTCDTR_D', y='ORIGAVTW', xlabel='', ylabel='Original Average Track Width')
    #PlotDistributionDiagramForProperty(vehiclePropery,  x='WGTCDTR_D', y='WHEELBAS', xlabel='', ylabel='Original Wheelbase')
    #PlotDistributionDiagramForProperty(vehiclePropery,  x='WGTCDTR_D', y='RATWGT', xlabel='', ylabel='Ratio Inflation Factor')
    StatisticFigure(vehiclePropery,  x='WGTCDTR_D', y='ORIGAVTW')
    StatisticFigure(vehiclePropery,  x='WGTCDTR_D', y='WHEELBAS')
    StatisticFigure(vehiclePropery,  x='WGTCDTR_D', y='RATWGT')

def StratifyData(dataInDict, func, min=0, max=100, bins=50):
    retDict = {}
    rang = (max-min)/bins    
    for key,value in dataInDict.items():
        currMin = min
        currMax = min + rang
        retDict[key] = {}
        for i in range(bins):            
            v = value[np.logical_and(value>currMin, value<= currMax)]
            retDict[key]['{0} - {1}'.format(currMin, currMax)] = func(v)
            currMin = currMax
            currMax = currMax + rang

    return retDict;

def CalculateEstimationForFeatureStratify(stratifyData, car, lt, ht, car_LT_ratio, car_HT_ratio, LT_car_ratio, LT_HT_ratio, HT_car_ratio, HT_LT_ratio, car_offset, LT_offset, HT_offset):
    carEst = 'Car Est'
    LTEst = 'Light Truck Est'
    HTEst = 'Heavy Truck Est'
    betterLT = '{0} : LT'
    betterHT = '{0} : HT'
    betterCar = '{0} : Car'
    stratifyData[carEst] = {}
    stratifyData[LTEst] = {}
    stratifyData[HTEst] = {}

    stratifyData[betterLT.format(carEst)] = {}
    stratifyData[betterHT.format(carEst)] = {}
    stratifyData[betterCar.format(LTEst)] = {}
    stratifyData[betterHT.format(LTEst)]= {}
    stratifyData[betterLT.format(HTEst)] = {}
    stratifyData[betterCar.format(HTEst)] = {}

    # Because the len of all 3 differnt type is the same, so we can only iterate once and will calculate all items.
    for key, item in stratifyData[car].items():
        if stratifyData[car][key] == stratifyData[car][key] and stratifyData[lt][key] == stratifyData[lt][key] and stratifyData[ht][key] == stratifyData[ht][key]:
            stratifyData[carEst][key] = car_LT_ratio * stratifyData[lt][key] + car_HT_ratio * stratifyData[ht][key] + car_offset
            stratifyData[LTEst][key] = LT_car_ratio * stratifyData[car][key] + LT_HT_ratio * stratifyData[ht][key] + LT_offset
            stratifyData[HTEst][key] = HT_LT_ratio * stratifyData[lt][key] + HT_car_ratio * stratifyData[car][key] + HT_offset

            stratifyData[betterLT.format(carEst)][key] = stratifyData[carEst][key] - stratifyData[car][key] - (stratifyData[lt][key] - stratifyData[carEst][key])
            stratifyData[betterHT.format(carEst)][key] = stratifyData[carEst][key] - stratifyData[car][key] - (stratifyData[ht][key] - stratifyData[carEst][key])

            stratifyData[betterCar.format(LTEst)][key] = stratifyData[LTEst][key] - stratifyData[lt][key] - (stratifyData[car][key] - stratifyData[lt][key])
            stratifyData[betterHT.format(LTEst)][key] = stratifyData[LTEst][key] - stratifyData[lt][key] - (stratifyData[ht][key] - stratifyData[lt][key])

            stratifyData[betterLT.format(HTEst)][key] = stratifyData[HTEst][key] - stratifyData[ht][key] - (stratifyData[lt][key] - stratifyData[ht][key])
            stratifyData[betterCar.format(HTEst)][key] = stratifyData[HTEst][key] - stratifyData[ht][key] - (stratifyData[car][key] - stratifyData[ht][key])
        else:
            stratifyData[carEst][key] =  np.nan
            stratifyData[LTEst][key] =   np.nan
            stratifyData[HTEst][key] =   np.nan

            stratifyData[betterLT.format(carEst)][key] =  np.nan
            stratifyData[betterHT.format(carEst)][key] =  np.nan
            stratifyData[betterCar.format(LTEst)][key] =  np.nan
            stratifyData[betterHT.format(LTEst)][key] =  np.nan
            stratifyData[betterLT.format(HTEst)][key] =  np.nan
            stratifyData[betterCar.format(HTEst)][key] =  np.nan

    return carEst, LTEst, HTEst, stratifyData

'''
Function to test estimation with selected data
'''
def ValidateSelectedData():
    SelData = pd.read_csv('selected_result.csv')
    
    # Remove the 0s from the data, we need all the necessary data for given accident type. If there are no full data for accident type, then we need to remove the cases.
    ReplaceZeroToNanRows(SelData, 'PassengerCar_Count')
    ReplaceZeroToNanRows(SelData, 'LightTruck_Count')
    ReplaceZeroToNanRows(SelData, 'HeavyTruck_Count')

    nonEmptyAccType = []
    uniAccType = pd.unique(SelData['Accident Type'])
    for ac in uniAccType:
        accType = SelData.loc[SelData['Accident Type'] == ac]   
        nullSum = accType.isnull().sum()
        #print('Accident Type {0}: {1}, null: {2}, totalnull: {3}'.format(ac, len(accType), nullSum, nullSum.sum()))

        # As only 3 columns are replaced 0 with Nan, and there are total 8 MAIS per accident type, so it will be 24 if all 3 are nan.
        if(nullSum.sum() < 24): 
            nonEmptyAccType.append(ac)
    print(nonEmptyAccType)

    SelectedData = SelData[SelData['Accident Type'].isin(nonEmptyAccType)]
    #SelectedData.to_csv('cleaned_csv.csv')
    print(SelectedData)
    #SelData.dropna(subset=['PassengerCar_Count', 'LightTruck_Count', 'HeavyTruck_Count'], inplace=True)
    ##SelData.to_csv('cleaned_csv.csv')
    #uniAccType = pd.unique(SelData['Accident Type'])
    #for ac in uniAccType:
    #    print('Accident Type {0}: {1}'.format(ac, len(SelData.loc[SelData['Accident Type'] == ac])))
    #print(uniAccType)

def CoefficientValidation():
    vehiclePropery = pd.read_csv('vehcileProperies.csv')
    car = 'Passenger Vehicle'
    lt = '6000 lbs & Under'
    ht = '6001-10000 Lbs'

    avTwFullList = ExtractVehiclePropertyForVehicleType(vehiclePropery)
    wheelbaseFullist = ExtractVehiclePropertyForVehicleType(vehiclePropery, y='WHEELBAS')
    
    stratifyDataAVT = StratifyData(avTwFullList, np.mean, min=100, max=190, bins=20)
    stratifyDataWhl = StratifyData(wheelbaseFullist, np.mean, min=200, max=450, bins=20)

    car_LT_ratio = 0.9
    car_HT_ratio = 0.1

    LT_car_ratio = 0.9
    LT_HT_ratio  = 0.1

    HT_car_ratio = 0.5
    HT_LT_ratio = 0.5

    car_offsetAvt = -3
    lt_offsetAvt = -0.36
    ht_offsetAvt = -15.5

    car_offsetWB = -20
    lt_offsetWB = -8.5
    ht_offsetWB = -58.7

    #car_offsetAvt = 0
    #lt_offsetAvt = 0
    #ht_offsetAvt = 0

    #car_offsetWB = 0
    #lt_offsetWB = 0
    #ht_offsetWB = 0

    carEst, LTEst, HTEst, calculatedDataAVT = CalculateEstimationForFeatureStratify(stratifyDataAVT, car, lt, ht, 
                                                                                 car_LT_ratio, car_HT_ratio, 
                                                                                 LT_car_ratio, LT_HT_ratio, 
                                                                                 HT_car_ratio, HT_LT_ratio,
                                                                                 car_offsetAvt, lt_offsetAvt, ht_offsetAvt)

    carEst, LTEst, HTEst, calculatedDataWB = CalculateEstimationForFeatureStratify(stratifyDataWhl, car, lt, ht, 
                                                                                 car_LT_ratio, car_HT_ratio, 
                                                                                 LT_car_ratio, LT_HT_ratio, 
                                                                                 HT_car_ratio, HT_LT_ratio,
                                                                                 car_offsetWB, lt_offsetWB, ht_offsetWB)

    #list = [value for key, value in calculatedDataAVT.items() if ':' in key]
    wilcoxAvt = {}
    wilcoxWB = {}

    for key in calculatedDataAVT:
        if ':' in key:
            wilcoxAvt[key] = pd.DataFrame(calculatedDataAVT[key], index=[0])
            wilcoxWB[key] = pd.DataFrame(calculatedDataWB[key], index=[0])
            print(key)            
            wlist = wilcoxAvt[key].iloc[0].dropna()
            Wilcoxon_From_ComparedList(wlist)
            #print(wlist)
    

    print('haha')
     
if __name__ == '__main__':
    #EstimateCoefficient()
    #EstimationValidation()

    #ValidateSelectedData()
   
    CoefficientValidation()
