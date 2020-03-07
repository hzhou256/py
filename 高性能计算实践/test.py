import time 
import math, sys
import pp
import numpy
import matrixtool
import copy
import ctypes


def get_mat(file_name):
	with open(file_name, 'r') as f:
		lines = f.readlines()
	order = int(lines[0].strip())
	mat = [[int(x) for x in line.strip().split()] for line in lines[1:]]
	return order, mat

def gen_empty_matrix(row, column):
    final_matrix = []
    for i in range(0,row):
        final_matrix.append([])
        for j in range(0,column):
            final_matrix[i].append(0)
    return final_matrix

def get_determinant(input_matrix):
    return numpy.linalg.det(input_matrix)

def get_transposed_matrix_parallel(input_matrix, row_start, row_end):
    matrix_rows = len(input_matrix)
    matrix_columns = len(input_matrix[0])
    final_matrix = gen_empty_matrix(matrix_columns,matrix_rows)
    for i in range(row_start, row_end):
        for j in range(0,matrix_columns):
            final_matrix[j][i] = input_matrix[i][j]
    return final_matrix

    
def get_cofactor_matrix_parallel(input_matrix, row_start, row_end):
    det = get_determinant(input_matrix)
    matrix_length = len(input_matrix)
    final_matrix = gen_empty_matrix(matrix_length,matrix_length)
    for i in range(row_start, row_end):
        newmatrix = []
        ur = input_matrix[0:i]
        if ur != []:
            urdeep = copy.deepcopy(ur)
            for row in urdeep:
                newmatrix.append(row)
        lr = input_matrix[i+1:]
        if lr != []:
            lrdeep = copy.deepcopy(lr)
            for row in lrdeep:
                newmatrix.append(row)
        for j in range(0,matrix_length):
            realmatrix = copy.deepcopy(newmatrix)
            for k in range(0, matrix_length -1):
                del realmatrix[k][j]
            final_matrix[i][j] = det * ((-1)**(i+j)) * get_determinant(realmatrix)
            #print(final_matrix[i][j])
    return final_matrix

def get_adjunct_matrix(input_matrix, row_start, row_end):
    return get_transposed_matrix_parallel(get_cofactor_matrix_parallel(input_matrix, row_start, row_end), row_start, row_end)

def get_inverse_matrix_parallel(input_matrix, row_start, row_end):
    return numpy.array(get_adjunct_matrix(input_matrix, row_start, row_end))



order, A = get_mat('E:/高性能计算实践/test.txt')
print(order)

#串行代码
print("{beg} serial process {beg}".format(beg='-'*16))
startTime = time.perf_counter()

B = matrixtool.get_inverse_matrix(A)
#print(B)

endTime = time.perf_counter()
print("use: %.3fs"%(endTime-startTime))


#并行代码
print("{beg} parallel process {beg}".format(beg='-'*16))
startTime = time.perf_counter()

job_server = pp.Server()
inputs = ((A, 0, 24), (A, 25, 49), (A, 50, 74), (A, 75, 99))

f1 = job_server.submit(get_inverse_matrix_parallel, (A, 0, 24), (gen_empty_matrix, get_determinant, get_transposed_matrix_parallel, get_cofactor_matrix_parallel, get_adjunct_matrix,),
        ("numpy", "copy", "sys"))
f2 = job_server.submit(get_inverse_matrix_parallel, (A, 25, 49), (gen_empty_matrix, get_determinant, get_transposed_matrix_parallel, get_cofactor_matrix_parallel, get_adjunct_matrix,),
        ("numpy", "copy", "sys"))
f3 = job_server.submit(get_inverse_matrix_parallel, (A, 50, 74), (gen_empty_matrix, get_determinant, get_transposed_matrix_parallel, get_cofactor_matrix_parallel, get_adjunct_matrix,),
        ("numpy", "copy", "sys"))
f4 = job_server.submit(get_inverse_matrix_parallel, (A, 75, 99), (gen_empty_matrix, get_determinant, get_transposed_matrix_parallel, get_cofactor_matrix_parallel, get_adjunct_matrix,),
        ("numpy", "copy", "sys"))

result = f1()+f2()+f3()+f4()
#print(result)
endTime = time.perf_counter()

print("use: %.3fs"%(endTime-startTime))

