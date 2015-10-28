import matplotlib.pyplot as plt
import os
import sys

inputfile = open('./testcase.loc')
lines = inputfile.readlines()
pos_loc = lines[0].strip().split(',')
neg_loc = lines[1].strip().split(',')

pos_loc = [float(item) for item in pos_loc]
neg_loc = [float(item) for item in neg_loc]

plt.axis([min([min(pos_loc), min(neg_loc)]), max([max(pos_loc), max(neg_loc)]), 0, 2])

y_pos_tag = [1 for item in pos_loc]
y_neg_tag = [1 for item in neg_loc]

plt.scatter(pos_loc, y_pos_tag, color = 'y')
plt.scatter(neg_loc, y_neg_tag, color = 'r')
plt.show()
