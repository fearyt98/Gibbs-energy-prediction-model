import os
import numpy as np
import pandas as pd
import traceback
from multiprocessing.pool import Pool
from sklearn import preprocessing
from pycalphad import Database

# modification of label encoding (sklearn) to run on multiple cores

# here must be path for calphad version database
db = Database('')

main_phases = np.unique(list(db.phases.keys()))

metals = (
        'LI','BE','NA','MG','AL','K','CA','SC','TI','V','CR','MN','FE',
        'CO','NI','CU','ZN','GA','Y','ZR','NB','MO','TC',
        'RH','PD','AG','CD','IN','SN','BA','LA','CE','PR','ND','PM','SM','EU',
        'GD','TB','DY','HO','ER','HF','TA','W','RE','OS','IR',
        'PT','AU','HG','TL','PB','BI', 'C', "SI")

if __name__ == "__main__":

    encoded_phases = pd.DataFrame(columns=['Encoded_phases'])

    data = pd.concat([
        pd.read_csv('test.csv', sep=',', names=['t','materials','conc','G','NP', 'Phases']), 
        pd.read_csv('test.csv', sep=',', names=['t','materials','conc','G','NP', 'Phases'])
    ])

    main_phases = np.append(main_phases, '')
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(main_phases)
    phases_array = np.zeros(shape=(len(data['Phases'])), dtype=np.ndarray)

    for id, item in enumerate(data['Phases']):
       phases_array[id] = np.asarray(item.replace('[','').replace(']','').replace("'","").replace(" ","").split(",")).astype(str)

    phases_encoded_array = []

    try:
        with Pool(processes=os.cpu_count()-2) as pool:
            for result in pool.map(label_encoder.transform, iterable=phases_array):
                phases_encoded_array.append(result)
        df_temp = pd.DataFrame(phases_encoded_array, columns=['Enc_ph_1', 'Enc_ph_2', 'Enc_ph_3'])
        data = pd.concat([data, df_temp], axis=1)
        data.to_csv('test.csv', mode='a', index=False, header=False)
    except Exception as e:
        print(traceback.format_exc())
    finally:
       pool.close()
       pool.join()