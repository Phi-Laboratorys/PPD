import os
import pandas as pd

types = 'Transmission_'

files_xy = [x for x in os.listdir() if '.xy' in x]
files_xls = [x for x in os.listdir() if '.xls' in x]

for f in files_xy:
    folder = f.split('_')[0]
    number = f.split('_')[2].split('.')[0]
    
    filename = types + folder + '_0' + number + '.csv'
    
    os.rename(f, folder + '/' + filename)

for f in files_xls:
    folder = f.split('_')[0].split('Dicke')[1]
    
    filename = types + folder + '_Fit' + '.xls'
    
    df = pd.read_excel(f, index_col=0, header=None)
    
    df.to_csv(folder + '/' + types + folder + '_Fit' + '.csv', index=False, header=False)
    
    os.rename(f, folder + '/' + filename)