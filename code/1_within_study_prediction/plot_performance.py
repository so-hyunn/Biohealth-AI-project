import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

dataset = "VanDenEnde"
output_path = "../../result/1_within_study_prediction/"

## visualize LOOCV ============================================================
LOOCV_perf = pd.read_csv(os.getcwd() + "/../../result/1_within_study_prediction/" + dataset + "/LOOCV_LogisticRegression_predictProba_False.txt",
                        sep="\t", index_col=1)

LOOCV_perf_re = LOOCV_perf.reindex(['NetBio', 'PD1', 'PD-L1', 'CTLA4', 'PD1_CTLA4', 'PD1_PD-L1_CTLA4', 'CD8T1',
                                'T_exhaust_Pos', 'CAF1', 'TAM_M2_M1_Pos', 'all-TME-Bio'])

LOOCV_perf_re["color"] = "gray"
LOOCV_perf_re["color"].values[0] = "blue"
LOOCV_perf_re["color"].values[1:6] = "silver"

plt.figure(figsize=(16, 9))

sns.barplot(data=LOOCV_perf_re, x=LOOCV_perf_re.index , y="accuracy", palette=LOOCV_perf_re["color"])
sns.set(font_scale=2)
plt.title(dataset)
plt.ylabel("Accuracy")
plt.xticks(rotation=45)

plt.savefig(output_path + "/" + dataset + "_LOOCV_accuracy.png")

plt.figure(figsize=(16, 9))

sns.barplot(data=LOOCV_perf_re, x=LOOCV_perf_re.index , y="F1", palette=LOOCV_perf_re["color"])
sns.set(font_scale=2)
plt.title(dataset)
plt.ylabel("F1")
plt.xticks(rotation=45)

plt.savefig(output_path + "/" + dataset + "_LOOCV_F1score.png")



## visualize monte carlo cv ===================================================
MCCV_perf = pd.read_csv(os.getcwd() + "/../../result/1_within_study_prediction/" + dataset + "/monteCarlo_testsize_0.2_LogisticRegression.txt",
                        sep="\t", index_col=3)

MCCV_perf["color"] = "gray"
MCCV_perf["color"].loc["NetBio"] = "blue"
MCCV_perf["color"].loc['PD1'] = "silver"
MCCV_perf["color"].loc['PD-L1'] = "silver"
MCCV_perf["color"].loc['CTLA4'] = "silver"
MCCV_perf["color"].loc['PD1_CTLA4'] = "silver"
MCCV_perf["color"].loc['PD1_PD-L1_CTLA4'] = "silver"

plt.figure(figsize=(16, 9))

sns.boxplot(data=MCCV_perf, x=MCCV_perf.index , y="accuracy", order=['NetBio', 'PD1', 'PD-L1', 'CTLA4', 'PD1_CTLA4',
            'PD1_PD-L1_CTLA4', 'CD8T1', 'T_exhaust_Pos', 'CAF1', 'TAM_M2_M1_Pos', 'all-TME-Bio'], orient="v",
            palette=MCCV_perf["color"])
sns.set(font_scale=2)
plt.title(dataset)
plt.ylabel("Accuracy")
plt.xticks(rotation=45)

plt.savefig(output_path + "/" + dataset + "_MonteCarlo_accuracy.png")


plt.figure(figsize=(16, 9))

sns.boxplot(data=MCCV_perf, x=MCCV_perf.index , y="F1", order=['NetBio', 'PD1', 'PD-L1', 'CTLA4', 'PD1_CTLA4',
            'PD1_PD-L1_CTLA4', 'CD8T1', 'T_exhaust_Pos', 'CAF1', 'TAM_M2_M1_Pos', 'all-TME-Bio'], orient="v",
            palette=MCCV_perf["color"])
sns.set(font_scale=2)
plt.title(dataset)
plt.ylabel("F1")
plt.xticks(rotation=45)

plt.savefig(output_path + "/" + dataset + "_MonteCarlo_F1score.png")
