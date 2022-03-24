import numpy as np


precisions = np.array(
                    [0.19620253, 0.38137083, 0.65555556, 0.81179423, 
                     0.88598901,0.93198263, 0.95386905, 0.9695586, 
                     0.98397436, 1.])

recalls = np.array([0.99237805, 0.99237805, 0.98932927, 0.98628049, 
                    0.98323171, 0.98170732, 0.97713415, 0.97103659, 
                    0.93597561, 0.])

recall_levels = np.linspace(0, 1.0, 11)
p = np.zeros_like(recall_levels)

for i,r_lvl in enumerate(recall_levels):
        r = [j for j in range(len(recalls)) if recalls[j] >= r_lvl]
        if len(r) > 0:
            p[i] = max(precisions[r])
        else:
            p[i] = 0
        print(f"r = {r}\n p = {p}\n\n")
        
print(f"Average precision: {np.sum(p)/len(p)}")