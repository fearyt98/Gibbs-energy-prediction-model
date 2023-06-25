import numpy as np
import pandas as pd
import itertools as itr
from pycalphad import Database, equilibrium
from pycalphad import variables as v
import os
import traceback
from multiprocessing.pool import Pool

# modification for data generation of ternary systems on multiple cores

w_step = 0.1
w_start = 0
w_end = 1
t_step = 20.15
t_start = 298.15
t_end = 3000
P = 101325
N = 1

# here must be path for calphad version database
db = Database('')

metals = (
        'LI','BE','NA','MG','AL','K','CA','SC','TI','V','CR','MN','FE',
        'CO','NI','CU','ZN','GA','Y','ZR','NB','MO','TC',
        'RH','PD','AG','CD','IN','SN','BA','LA','CE','PR','ND','PM','SM','EU',
        'GD','TB','DY','HO','ER','HF','TA','W','RE','OS','IR',
        'PT','AU','HG','TL','PB','BI', 'C', "SI")

main_phases = list(db.phases.keys())
main_elements = list(db.elements)

# similar structure to the structure in data_generation.ipyng 
def ternary_gb_en(t_local_start, t_local_step, main_eq, main_met, conc_met_f, conc_met_s, w_conc_f, w_conc_s):
    dataframe = pd.DataFrame(columns=['t','materials','conc','G','NP', 'Phases']) 
    t_temp = t_local_start 
    for id_t, item in enumerate(main_eq.GM[0][0]):
        dataframe.loc[len(dataframe)] = [
            t_temp, 
            (main_met, conc_met_f, conc_met_s), 
            [w_conc_f, w_conc_s],
            item[0][0].values.tolist(), 
            main_eq.NP[0][0][id_t][0][0].values.tolist(), 
            main_eq.Phase[0][0][id_t][0][0].values.tolist()]
        t_temp = t_temp + t_local_step
    return dataframe

# factory function to implement the ability to work with data on multiple cores 
# and formatting into the required form to generate Gibbs energy data
def factory_ternary(w_conc_f, w_conc_s, main_met, conc_met_f, conc_met_s):
    print((main_met, conc_met_f, conc_met_s, w_conc_f, w_conc_s))
    elem = [main_met, conc_met_f, conc_met_s]
    cond = {v.X(conc_met_f):(w_conc_f), v.X(conc_met_s):(w_conc_s), v.T:(t_start, t_end, t_step), v.P:P, v.N: N}
    main_eq = equilibrium(db, elem, main_phases, cond, output="GM")
    return ternary_gb_en(t_start, t_step, main_eq, main_met, conc_met_f, conc_met_s, w_conc_f, w_conc_s)
    
if __name__  == "__main__":

    combinations = itr.combinations(metals, 3)
    accuracy = 1
    conc = np.arange(w_start, w_end, w_step)
    w = itr.product(conc, conc) 

    conc_array = []
    for i, k in w:
        if i+k < 1 and i !=0 and k !=0: conc_array.append((np.round_(i, accuracy),np.round_(k, accuracy)))
    conc_array = np.asarray(conc_array)

    for i, j, k in itr.islice(combinations, 0, None):
        
        df_temp = pd.DataFrame(columns=['t','materials','conc','G','NP', 'Phases'])
        
        # array to implement the ability to work on multiple cores
        pl_array = np.empty(shape=(len(conc_array), 5), dtype=tuple)
        for id, item in enumerate(conc_array):
            pl_array[id][0] = item[0]
            pl_array[id][1] = item[1]
            pl_array[id][2] = i
            pl_array[id][3] = j
            pl_array[id][4] = k

        try:
            with Pool(processes=os.cpu_count()-2) as pool:
                for result in pool.starmap(factory_ternary, pl_array):
                    df_temp = pd.concat([df_temp, result])
            df_temp.to_csv('test.csv', mode='a', index=False, header=False)
            del df_temp  
        except Exception as e:
            print(traceback.format_exc())
            with open('gb_errors_ternary.txt', 'a') as errors:
                errors.write(str([i, j, k]))
        finally:
            pool.close()
            pool.join()

        with open('last_mets_ternary.txt', 'w') as last:
            last.write(str([i, j, k]))