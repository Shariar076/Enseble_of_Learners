import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
# performance_history = np.loadtxt('perf_history_all.txt')
name = 'mfcc'
# perf_id_knn = np.loadtxt('./Results/'+name+'/density with knn/perf_history_all.txt')
# perf_id_rf = np.loadtxt('./Results/'+name+'/density with rf/perf_history_all.txt')
# perf_en = np.loadtxt('./Results/'+name+'/Entropy/perf_history_all.txt')
# perf_mar = np.loadtxt('./Results/'+name+'/Margin/perf_history_all.txt')
# perf_moe = np.loadtxt('./Results/'+name+'/MoE/perf_history_all.txt')
# perf_rbmal = np.loadtxt('./Results/'+name+'/RBMAL/perf_history_all.txt')
# perf_us = np.loadtxt('./Results/'+name+'/Us/perf_history_all.txt')


name = 'alpha/checkerboard_rotated'
perf_id_knn = np.loadtxt('./Results/'+name+'/1/perf_history_all.txt')
perf_id_rf = np.loadtxt('./Results/'+name+'/5/perf_history_all.txt')
perf_en = np.loadtxt('./Results/'+name+'/10/perf_history_all.txt')
perf_mar = np.loadtxt('./Results/'+name+'/15/perf_history_all.txt')
# perf_moe = np.loadtxt('./Results/'+name+'/MoE/perf_history_all.txt')
# perf_rbmal = np.loadtxt('./Results/'+name+'/RBMAL/perf_history_all.txt')
# perf_us = np.loadtxt('./Results/'+name+'/Us/perf_history_all.txt')


#creditcard
# perf_moe = perf_moe[:,:101]

# ID_knn = np.mean(perf_id_knn,axis=0)
# ID_rf = np.mean(perf_id_rf,axis=0)
# Us_En = np.mean(perf_en,axis=0)
# MoE = np.mean(perf_moe,axis=0)
# RBMAL = np.mean(perf_rbmal,axis=0)
# Us_M = np.mean(perf_mar,axis=0)
# Us_LC = np.mean(perf_us,axis=0)

win = 21
deg = 8
ID_knn = savgol_filter(np.mean(perf_id_knn,axis=0),win,deg)
ID_rf = savgol_filter(np.mean(perf_id_rf,axis=0),win,deg)
Us_En = savgol_filter(np.mean(perf_en,axis=0),win,deg)
Us_M = savgol_filter(np.mean(perf_mar,axis=0),win,deg)
# Us_LC = savgol_filter(np.mean(perf_us,axis=0),win,deg)
# MoE = savgol_filter(np.mean(perf_moe,axis=0),win,deg)
# RBMAL = savgol_filter(np.mean(perf_rbmal,axis=0),win,deg)

# sft = ID_knn[0]-MoE[0]
# MoE+= sft
# sft = ID_knn[0]-ID_rf[0]
# ID_rf+= sft
# sft = ID_knn[0]-RBMAL[0]
# RBMAL+= sft
# sft = ID_knn[0]-Us_En[0]
# Us_En+= sft
# sft = ID_knn[0]-Us_LC[0]
# Us_LC+= sft
# sft = ID_knn[0]-Us_M[0]
# Us_M+= sft




# x = range(1001)
# ar = np.zeros(1001)
# ar[300:1001]=ID_knn[0:]
# plt.plot(x[300:],ar[300:], label= 'ID_knn',color= 'b')
# ar[300:1001]=ID_rf[0:]
# plt.plot(x[300:],ar[300:], label= 'ID_rf',color= 'g')
# ar[300:1001]=MoE[0:]
# plt.plot(x[300:],ar[300:], label= 'EoL',color= 'm')
# ar[300:1001]=RBMAL[0:]
# plt.plot(x[300:],ar[300:], label= 'RBMAL',color= 'y')
# ar[300:1001]=Us_En[0:]
# plt.plot(x[300:],ar[300:], label= 'Us-En',color= 'r')


# plt.plot(ID_knn, label= 'ID_knn',color= 'b')
# plt.plot(ID_rf, label= 'ID_rf',color= 'g')
# plt.plot(MoE, label= 'EoL',color= 'm')
# plt.plot(RBMAL, label= 'RBMAL',color= 'y')
# plt.plot(Us_En, label= 'Us_En',color= 'r')
# plt.plot(Us_M, label= 'Us_M',color= 'c')
# plt.plot(Us_LC, label= 'Us_LC',color= 'coral')
# plt.plot(performance_history[1], label= 'label2')
# plt.xlabel('# labelled points')
# plt.ylabel('accuracy')
# plt.legend(loc='lower right')
# plt.plot('curve_of_learner.png')
# plt.savefig('./Results/'+name+'.png')
# plt.show()


plt.plot(ID_knn, label= 'α = 0.1',color= 'b')
plt.plot(ID_rf, label= 'α = 0.5',color= 'g')
plt.plot(Us_En, label= 'α = 1.0',color= 'r')
plt.plot(Us_M, label= 'α = 1.5',color= 'c')
plt.xlabel('# labelled points')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.savefig('./Results/'+name+'.png')
plt.show()
