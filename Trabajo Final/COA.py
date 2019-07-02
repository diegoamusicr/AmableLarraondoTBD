import random
import numpy as np
import sys
from K_clustering import *

orig_stdout = sys.stdout
f = open('out.txt', 'w')
sys.stdout = f

def f1(x):
	return sum(pow(x, 2))

class COA:

	def __init__(self, f, dimension, var_low, var_high, operation='min', init_pop=10, max_pop=20, egg_low=2, egg_high=4, alpha=3, eggs_kill_ratio=0.1, deviation=np.pi/6, generations=200, clusters_num=3):
		self.f = f
		self.dimension = dimension
		self.var_low = var_low
		self.var_high = var_high
		self.operation = operation
		self.init_pop = init_pop
		self.max_pop = max_pop
		self.egg_low = egg_low
		self.egg_high = egg_high
		self.alpha = alpha
		self.eggs_kill_ratio = eggs_kill_ratio
		self.deviation = deviation
		self.cluster = K_Cluster(clusters_num)
		self.generations = generations

	def CalcProfit(self, individuals):
		if self.operation == 'min':
			return -np.apply_along_axis(self.f, 1, individuals)
		return np.apply_along_axis(self.f, 1, individuals)

	def CreateRandomLocations(self, size):
		return np.array([[np.random.uniform(self.var_low, self.var_high) for j in range(self.dimension)] for i in range(size)])

	def InitPob(self):
		self.cuckoos = self.CreateRandomLocations(self.init_pop)
		self.cuckoos_profit = self.CalcProfit(self.cuckoos)

	def CalcEggs(self):
		self.num_eggs = np.array([np.random.randint(self.egg_low, self.egg_high+1) for i in range(len(self.cuckoos))])
		self.ELR = self.alpha * self.num_eggs / np.sum(self.num_eggs) * (self.var_high - self.var_low)

	def LayEggs(self):
		self.eggs = np.array([]).reshape(0, self.dimension)
		for i in range(len(self.cuckoos)):
			angles = np.random.random(self.num_eggs[i]) * 2 * np.pi
			radius = self.ELR[i] * np.sqrt(np.random.random(self.num_eggs[i]))
			addValues = radius * np.cos(angles) + radius * np.sin(angles)
			eggs = np.array([self.cuckoos[i] + addValues[j] for j in range(self.num_eggs[i])])
			self.eggs = np.concatenate((self.eggs, eggs))
		self.eggs[self.eggs < self.var_low] = self.var_low
		self.eggs[self.eggs > self.var_high] = self.var_high

	def KillAndGrowEggs(self):
		eggs_profit = self.CalcProfit(self.eggs)
		profit_order = eggs_profit.argsort()[::-1]
		eggs_to_kill = int(np.around(len(self.eggs) * self.eggs_kill_ratio))

		self.eggs = self.eggs[profit_order]
		self.eggs = self.eggs[:-eggs_to_kill]
		eggs_profit = eggs_profit[profit_order]
		eggs_profit = eggs_profit[:-eggs_to_kill]

		if len(self.eggs) > self.max_pop:
			self.cuckoos = self.eggs[:self.max_pop]
			self.cuckoos_profit = eggs_profit[:self.max_pop]
		else:
			self.cuckoos = self.eggs
			self.cuckoos_profit = eggs_profit

	def ImmigrateCuckoos(self):
		groups = self.cluster.GetGroups(self.cuckoos)
		groups_profit = np.array([np.mean(self.CalcProfit(groups[i])) for i in range(len(groups))])
		best_group_arg = groups_profit.argmax()
		best_group_pos = self.cluster.CalcCentroid(groups[best_group_arg])
		for i in range(len(self.cuckoos)):
			diff = best_group_pos - self.cuckoos[i]
			magnitude = np.linalg.norm(diff) * np.random.random()
			angle = np.arccos(np.sum(best_group_pos * self.cuckoos[i]) / (np.linalg.norm(best_group_pos) * np.linalg.norm(self.cuckoos[i]))) + \
					np.random.uniform(-self.deviation, self.deviation)
			self.cuckoos[i] = self.cuckoos[i] + magnitude * np.cos(angle) + magnitude * np.sin(angle)
			#self.cuckoos[i] = self.cuckoos[i] + np.random.random() * (best_group_pos - self.cuckoos[i])
		self.cuckoos[self.cuckoos < self.var_low] = self.var_low
		self.cuckoos[self.cuckoos > self.var_high] = self.var_high
		self.cuckoos_profit = self.CalcProfit(self.cuckoos)

	def PrintCuckoos(self, cuckoos, profit):
		for i in range(len(cuckoos)):
			print (cuckoos[i], '----------', profit[i])

	def Run(self):
		self.InitPob()

		print ("Initial Cuckoos:")
		self.PrintCuckoos(self.cuckoos, self.cuckoos_profit)
		print ("------------------------------------------------------------------------")
		for i in range(self.generations):

			print("ITERATION %d:" % (i))

			self.CalcEggs()
			print ("Number of eggs: ", self.num_eggs)
			print ("ELR: ", self.ELR)

			self.LayEggs()
			print ("Eggs: ")
			print (self.eggs)

			self.KillAndGrowEggs()
			print ("Eggs (after killing worst):")
			self.PrintCuckoos(self.cuckoos, self.cuckoos_profit)

			self.ImmigrateCuckoos()
			print ("Cuckoos after immigration:")
			self.PrintCuckoos(self.cuckoos, self.cuckoos_profit)

			print("------------------------------------------------------------------------ END ITERATION %d" % (i))

		
		print("************************************************************************")
		print("FINAL RESULT (%d generations):"%(self.generations))

		print("Final Cuckoos:")
		self.PrintCuckoos(self.cuckoos, self.cuckoos_profit)

		print("************************************************************************")

C = COA(f1, 2, -10, 10, generations=100)
C.Run()

sys.stdout = orig_stdout
f.close()