import copy
import ctypes
import random
import numpy


def gen_empty_matrix(row, column):
    final_matrix = []
    for i in range(0,row):
        final_matrix.append([])
        for j in range(0,column):
            final_matrix[i].append(0)
    return final_matrix

def get_determinant(input_matrix):
    return numpy.linalg.det(input_matrix)

def get_transposed_matrix(input_matrix):
    matrix_rows = len(input_matrix)
    matrix_columns = len(input_matrix[0])
    final_matrix = gen_empty_matrix(matrix_columns,matrix_rows)
    for i in range(0,matrix_rows):
        for j in range(0,matrix_columns):
            final_matrix[j][i] = input_matrix[i][j]
    return final_matrix

def get_cofactor_matrix(input_matrix):
    matrix_length = len(input_matrix)
    final_matrix = gen_empty_matrix(matrix_length,matrix_length)
    for i in range(0,matrix_length):
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
            final_matrix[i][j] = ((-1)**(i+j)) * get_determinant(realmatrix)
    return final_matrix

def get_adjunct_matrix(input_matrix):
    return get_transposed_matrix(get_cofactor_matrix(input_matrix))

def multiply_matrix_by_scalar(input_matrix, scalar):
    matrix_rows = len(input_matrix)
    matrix_columns = len(input_matrix[0])
    final_matrix = gen_empty_matrix(matrix_rows,matrix_columns)
    for i in range(0,matrix_rows):
        for j in range(0,matrix_columns):
            final_matrix[i][j] = (input_matrix[i][j]) * scalar
    return final_matrix
    
                
def get_inverse_matrix(input_matrix):
    det = numpy.linalg.det(input_matrix)
    return numpy.array(multiply_matrix_by_scalar(get_adjunct_matrix(input_matrix),1/det))
