import matplotlib.pyplot as plt
import plot
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit

def linear(x, a, b):
    return a * x + b

#Obacht!! Paul-spezifischer Pfad!!
data_folder = "/Users/paul/Documents/GitHub/PPD/Versuch_DF/Daten"
pic_folder = "/Users/paul/Documents/GitHub/PPD/Versuch_DF/Bilder/Auswertung/43"

plot.parameters(True, 30, (16,8), 100, colorblind = False)



dfReflexInput = pd.read_csv(data_folder + "/DickeReflex.csv")
dfTransInput = pd.read_csv(data_folder + "/DickeTrans.csv")
dfBothInput = dfReflexInput.copy()
dfTransCopy = dfTransInput.copy()
dfBothInput.append(dfTransCopy)

cList = [300, 250, 200, 150, 100, 50, 25]

mList = []
steList = []
for c in cList:
    dfReflexSelect= dfReflexInput.copy()
    dfReflexSelect = dfReflexSelect[dfReflexSelect['c (mg/ml)'] == c]
    m = np.mean(dfReflexSelect['d (nm)'])
    ste = np.std(dfReflexSelect['d (nm)'])#/((dfReflexSelect.shape[0])**0.5)
    mList.append(m)
    steList.append(ste)
dfReflex = pd.DataFrame(data={'c':cList,'d':mList,'dE':steList})

mList = []
steList = []
for c in cList:
    dfTransSelect= dfTransInput.copy()
    dfTransSelect = dfTransSelect[dfTransSelect['c (mg/ml)'] == c]
    m = np.mean(dfTransSelect['d (nm)'])
    ste = np.std(dfTransSelect['d (nm)'])#/((dfTransSelect.shape[0])**0.5)
    mList.append(m)
    steList.append(ste)
dfTrans = pd.DataFrame(data={'c':cList,'d':mList,'dE':steList})

mList = []
steList = []
for c in cList:
    dfBothSelect = dfBothInput.copy()
    dfBothSelect = dfBothSelect[dfBothSelect['c (mg/ml)'] == c]
    m = np.mean(dfBothSelect['d (nm)'])
    ste = np.std(dfBothSelect['d (nm)'])#/((dfBothSelect.shape[0])**0.5)
    mList.append(m)
    steList.append(ste)
dfBoth = pd.DataFrame(data={'c':cList,'d':mList,'dE':steList})

dfFit1 = dfBoth.copy()
dfFit1 = dfFit1[dfBoth['c'] < 150]

dfFit2 = dfBoth.copy()
dfFit2 = dfFit2[dfBoth['c'] > 150]


popt1, var1 = curve_fit(linear, dfFit1['c'], dfFit1['d'])
popt2, var2 = curve_fit(linear, dfFit2['c'], dfFit2['d'])
    
a1, b1 = popt1
a2, b2 = popt2

print(a1, b1)
print(a2, b2)
    
fig = plt.figure()
#plt.errorbar(dfReflex['c'], dfReflex['d'], yerr=dfReflex['dE'], label='Reflexion')
#plt.errorbar(dfTrans['c'], dfTrans['d'], yerr=dfTrans['dE'], label='Transmission')
plt.errorbar(dfBoth['c'], dfBoth['d'], yerr=dfBoth['dE'], fmt='rx', label='Mittelwert beider Messmethoden')
plt.errorbar([0, 200], [linear(0,a1,b1), linear(200,a1,b1)], label='Fit 1')
plt.errorbar([100, 300], [linear(100,a2,b2), linear(300,a2,b2)], label='Fit 2')
plt.xlabel("c in mg/ml")
plt.ylabel("d in nm")
    
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=8, frameon=False, columnspacing=0.5, handlelength=0.5)

#plt.savefig(pic_folder + "/Schichtdicken-Fit.pdf", bbox_inches='tight')
plt.show()