import sys
import matplotlib.pyplot as plt
from COA import *

class COA_Cluster:

	def __init__(self, data, k=2):
		self.data = data
		self.k = k

	def EuclideanDist(self, v1, v2):
		return np.sqrt(sum(pow(v1 - v2, 2)))/len(v1)

	def CalcSolutionFitness(self, groupstring):
		groupstring_rounded = np.around(groupstring).astype(int)
		distance_sum = 0.0
		for i in range(self.k):
			if i in groupstring_rounded:
				group = self.data[groupstring_rounded == i]
				center = group[np.random.choice(range(len(group)))]
				for j in range(len(group)):
					distance_sum += self.EuclideanDist(group[j], center)
		return distance_sum

	def CalcGroups(self):
		orig_stdout = sys.stdout
		f = open('out_LA.txt', 'w')
		sys.stdout = f

		C = COA(self.CalcSolutionFitness, len(self.data), 0, self.k-1, generations=250, alpha=1, clusters_num=2)
		C.Run()
		self.groups = np.around(C.GetBest()).astype(int)

		sys.stdout = orig_stdout
		f.close()

	def GetGroups(self):
		return self.groups

	def GetGroupedData(self):
		grouped_data = []
		for i in range(self.k):
			grouped_data += [self.data[self.groups == i]]
		return grouped_data

data = np.array([[  0.37173599,   2.30956196],
				 [ -1.8429661 ,   0.09485988],
				 [ -0.22269248,   1.7151335 ],
				 [ -0.63475995,  -0.53236636],
				 [ -2.41371211,  -2.31131852],
				 [ -4.66404446,  -2.72621849],
				 [  2.50606629,   4.44389226],
				 [ -5.05853108,  -3.12070511],
				 [ -0.63264744,   1.30517853],
				 [  4.6662476 ,   6.60407357],
				 [  1.82210852,   3.75993449],
				 [ -1.71123651,   0.22658946],
				 [ -0.89316855,   1.04465743],
				 [  1.85275329,   7.27782362],
				 [  0.81325359,   6.23832392],
				 [  1.57680599,   7.00187632],
				 [  5.32308382,   0.49643775],
				 [  3.6129266 ,  -1.21371948],
				 [  0.63726155,   6.06233188],
				 [  1.8462414 ,   7.27131173],
				 [ -1.41265967,  -1.31026608],
				 [ -3.39645217,  -3.29405858],
				 [ 10.        ,   5.19201617],
				 [  7.20531487,   2.3786688 ],
				 [  1.24625651,  -3.58038957],
				 [  6.45424276,   1.62759669],
				 [  9.74895952,   4.74335883],
				 [  6.56629003,   1.56068934],
				 [  5.31884088,   0.31324019],
				 [  3.91403109,  -1.09156959],
				 [  3.07849015,   5.01631612],
				 [  2.11589157,   4.05371755],
				 [ -0.76411638,   5.28773691],
				 [ -4.80217379,   1.24967949],
				 [ -7.40413454,  -1.35228125],
				 [  1.26797039,  -3.55867569],
				 [  0.05669265,  -4.76995343],
				 [ -5.99723841,  -2.31388197],
				 [-10.        ,  -6.99771785],
				 [ -8.96055249,  -5.27719604],
				 [ -7.33808082,  -3.65472437],
				 [ -1.42925936,   2.25409708],
				 [ -3.98543934,  -0.3020829 ],
				 [ -2.60025727,  -2.49786368],
				 [ -6.57685428,  -6.47446069],
				 [ -0.45504883,  -0.35265524],
				 [ -0.72677363,  -0.62438004],
				 [ 10.        ,  -0.46506595],
				 [ 10.        ,  -0.55023172],
				 [ -7.43310146,   6.05170653],
				 [ -5.93771232,   7.54709567],
				 [ -6.1407742 ,   7.34403379],
				 [ 10.        ,  -0.7914364 ],
				 [  7.12634073,  -5.20543224],
				 [  5.73711438,  -6.59465859],
				 [ 10.        ,  -2.24686096],
				 [ -2.72978296,   7.85333571],
				 [ -2.37630012,   8.20681855]])

C = COA_Cluster(data, 3)
C.CalcGroups()
g_data = C.GetGroupedData()
plt.plot(g_data[0][:,0], g_data[0][:,1], 'ro')
plt.plot(g_data[1][:,0], g_data[1][:,1], 'bo')
plt.plot(g_data[2][:,0], g_data[2][:,1], 'go')
plt.show()
