import numpy as np

def get_mat(file_name):
	with open(file_name, 'r') as f:
		lines = f.readlines()
	side = int(lines[0].strip())
	mat = [[int(x) for x in line.strip().split()] for line in lines[1:]]
	return side, mat

