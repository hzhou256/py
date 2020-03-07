import copy
import ctypes
import random
from fractions import Fraction

def input_matrix():
    correct_value_ms = False
    correct_value_ms2 = False
    while correct_value_ms == False:
        try:
            m = int(input("m x n matrix, m: "))
            if m >= 1:
                correct_value_ms = True
            else:
                print("Please input an integer equal or higher than 1.")
        except ValueError:
            correct_value_ms = False
            print("Please input a valid integer.")
    while correct_value_ms2 == False:
        try:
            n = int(input("m x n matrix, n: "))
            if n >= 1:
                correct_value_ms2 = True
            else:
                print("Please input an integer equal or higher than 1.")
        except ValueError:
            correct_value_ms2 = False
            print("Please input a valid integer.")
    om = []
    for i in range(0,m):
        current_row = []
        for j in range(0,n):
            correct_value_me = False
            cur_prompt = "Element " + str(i+1) + "," + str(j+1) + ": "
            while correct_value_me == False:
                try:
                    current_row.append(Fraction(input(cur_prompt)))
                    correct_value_me = True
                except ValueError:
                    correct_value_me = False
                    print("Please input a valid number.")
        om.append(current_row)
    return om
    
def validate_square_matrix(input_matrix):
    num_of_rows = len(input_matrix)
    for row in input_matrix:
        if len(row) != num_of_rows and num_of_rows != 0:
            return False
    return True

def gen_empty_matrix(row, column):
    final_matrix = []
    for i in range(0,row):
        final_matrix.append([])
        for j in range(0,column):
            final_matrix[i].append(Fraction(0))
    return final_matrix

def get_determinant(input_matrix):
    matrix_eval = validate_square_matrix(input_matrix)
    if matrix_eval == False:
        return False
    matrix_length = len(input_matrix)
    det = 0
    if matrix_length == 1:
        det = input_matrix[0][0]
    elif matrix_length == 2:
        det = -1*(input_matrix[0][1] * input_matrix[1][0]) + (input_matrix[0][0] * input_matrix[1][1])
    else:
        for i in range(0,matrix_length):
            tn = input_matrix[0][i]
            if tn != 0:
                newmatrix = copy.deepcopy(input_matrix[1:])
                for j in range(0,matrix_length-1):
                    del newmatrix[j][i]
                if i % 2 == 0:
                    det += Fraction(get_determinant(newmatrix)) * tn
                else:
                    det -= Fraction(get_determinant(newmatrix)) * tn
    return det

def get_transposed_matrix(input_matrix):
    matrix_rows = len(input_matrix)
    matrix_columns = len(input_matrix[0])
    final_matrix = gen_empty_matrix(matrix_columns,matrix_rows)
    for i in range(0,matrix_rows):
        for j in range(0,matrix_columns):
            final_matrix[j][i] = input_matrix[i][j]
    return final_matrix

def get_cofactor_matrix(input_matrix):
    matrix_eval = validate_square_matrix(input_matrix)
    if matrix_eval == False:
        return False
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
            final_matrix[i][j] = ((-1)**(i+j)) * Fraction(get_determinant(realmatrix))
    return final_matrix

def get_adjunct_matrix(input_matrix):
    matrix_eval = validate_square_matrix(input_matrix)
    if matrix_eval == False:
        return False
    return get_transposed_matrix(get_cofactor_matrix(input_matrix))

def multiply_matrix_by_scalar(input_matrix, scalar):
    matrix_rows = len(input_matrix)
    matrix_columns = len(input_matrix[0])
    final_matrix = gen_empty_matrix(matrix_rows,matrix_columns)
    for i in range(0,matrix_rows):
        for j in range(0,matrix_columns):
            final_matrix[i][j] = Fraction((input_matrix[i][j])) * Fraction(scalar)
    return final_matrix
    
def matrix_multiplication(input_matrix_1, input_matrix_2):
    matrix_1_rows = len(input_matrix_1)
    matrix_1_columns = len(input_matrix_1[0])
    matrix_2_rows = len(input_matrix_2)
    matrix_2_columns = len(input_matrix_2[0])
    if matrix_1_columns != matrix_2_rows:
        return False
    newmatrix = gen_empty_matrix(matrix_1_rows,matrix_2_columns)
    for m1i in range(0,matrix_1_rows):
        for m2j in range(0,matrix_2_columns):
            cv = 0
            for i in range(0, matrix_1_columns):
                cv += Fraction(input_matrix_1[m1i][i]) * Fraction(input_matrix_2[i][m2j])
            newmatrix[m1i][m2j] = cv
    return newmatrix
                
def get_inverse_matrix(input_matrix):
    matrix_eval = validate_square_matrix(input_matrix)
    if matrix_eval == False:
        return False
    det = get_determinant(input_matrix)
    if det != 0:
        if len(input_matrix) == len(input_matrix[0]) == 1:
            nmat = gen_empty_matrix(1,1)
            nmat[0][0] = Fraction(1/det)
            return nmat
        else:
            return multiply_matrix_by_scalar(get_adjunct_matrix(input_matrix),Fraction(1/det))
    else:
        return False

def matrix_addition(input_matrix_1,input_matrix_2):
    matrix_1_rows = len(input_matrix_1)
    matrix_1_columns = len(input_matrix_1[0])
    matrix_2_rows = len(input_matrix_2)
    matrix_2_columns = len(input_matrix_2[0])
    if matrix_1_columns == matrix_2_columns and matrix_1_rows == matrix_2_rows:
        final_matrix = gen_empty_matrix(matrix_1_rows,matrix_1_columns)
        for i in range(matrix_1_rows):
            for j in range(matrix_1_columns):
                final_matrix[i][j] = Fraction(input_matrix_1[i][j]) + Fraction(input_matrix_2[i][j])
        return final_matrix
    return False
    
def fraction_to_string(number):
    num, den = Fraction(number).numerator, Fraction(number).denominator
    if den == 1:
        return str(num)
    if den != 1:
        return str(num) + "/" + str(den)

def print_matrix(input_matrix):
    matrix_rows = len(input_matrix)
    matrix_columns = len(input_matrix[0])
    max_len = 0
    final_matrix = gen_empty_matrix(matrix_rows,matrix_columns)
    final_matrix_2 = gen_empty_matrix(matrix_rows,matrix_columns)
    for i in range(0,matrix_rows):
        for j in range(0,matrix_columns):
            try:
                string = fraction_to_string(input_matrix[i][j])
            except ValueError:
                string = str(input_matrix[i][j])
            orlen = len(string)
            if orlen > max_len:
                max_len = orlen
            final_matrix[i][j] = string
    for i in range(0,matrix_rows):
        for j in range(0,matrix_columns):
            string2 = final_matrix[i][j]
            while len(string2) < max_len+1:
                string2 = " " + string2
            final_matrix_2[i][j] = string2
    for row in final_matrix_2:
        outrow = "".join(row)
        print(outrow)
    return None
    
def purged_input(display):
    alpha = False
    while alpha == False:
        ri = str(input(display)).upper()
        if len(ri) == 1 and ri[0].isalpha() == True:
            alpha = True
        else:
            print("Please enter data again")
    return ri
    
def compare_matrices(input_matrix_1,input_matrix_2):
    matrix_1_rows = len(input_matrix_1)
    matrix_1_columns = len(input_matrix_1[0])
    matrix_2_rows = len(input_matrix_2)
    matrix_2_columns = len(input_matrix_2[0])
    if matrix_1_columns == matrix_2_columns and matrix_1_rows == matrix_2_rows:
        final_matrix = gen_empty_matrix(matrix_1_rows,matrix_1_columns)
        for i in range(matrix_1_rows):
            for j in range(matrix_1_columns):
                if Fraction(input_matrix_1[i][j]) == Fraction(input_matrix_2[i][j]):
                    final_matrix[i][j] = "OK"
                else:
                    final_matrix[i][j] = "ER"
        print_matrix(final_matrix)
    else:
        print("Matrices of different sizes")
    
def edit_matrix(input_matrix):
    print("Current matrix: ")
    print_matrix(input_matrix)
    print("To exit press X ")
    is_done = False
    while is_done == False:
        ci = False
        cj = False
        correct_value_me = False
        while ci == False:
            i = str(input("Element to edit (i,j) i: "))
            if i.isdecimal() == True and int(i) <= len(input_matrix):
                ci = True
                i = int(i)
            elif i.upper() == "X":
                ci = True
                is_done = True
            elif i.isdecimal() == True and int(i) > len(input_matrix):
                print("Element not in matrix")
            else:
                print("Only positive integers allowed")
        if is_done == True:
            return input_matrix
        while cj == False:
            j = str(input("Element to edit (i,j) j: "))
            if j.isdecimal() == True and int(j) <= len(input_matrix[0]):
                cj = True
                j = int(j)
            elif j.upper() == "X":
                cj = True
                is_done = True
            elif j.isdecimal() == True and int(j) > len(input_matrix[0]):
                print("Element not in matrix")
            else:
                print("Only positive integers allowed")
        if is_done == True:
            return input_matrix
        cur_prompt = "Enter element " + str(i) + "," + str(j) + ": "
        while correct_value_me == False:
            try:
                new_element = Fraction(input(cur_prompt))
                correct_value_me = True
            except ValueError:
                correct_value_me = False
                print("Please input a valid number.")
        input_matrix[i-1][j-1] = new_element
        print("Current matrix: ")
        print_matrix(input_matrix)
        
def matrix_rng(row, column, min, max):
    rng = random.SystemRandom()
    final_matrix = []
    for i in range(0,row):
        final_matrix.append([])
        for j in range(0,column):
            final_matrix[i].append(Fraction(rng.randint(min,max)))
    return final_matrix

matrix_dict = {}
want_to_quit = False
print("Matrix calculator r2.0")
print("By fabrizziop")
print("MIT Licence")
while want_to_quit == False:
    avail_matrix = sorted(list(matrix_dict.keys()))
    availstr = ""
    for item in avail_matrix:
        availstr += str(item) + " "
    print("Matrices in memory: ", availstr)
    action = (str(input("1: Input, 2: Delete, 3: Cofact, 4: Adj, 5: Inv, 6: Mult(s), 7: Mult(m), 8: Add, 9: Print, 10: Det, 11: Trans, 12: Compare, 13: Edit, 14: RNG, X: Exit "))).upper()
    if action == "1":
        letter = purged_input("Assign a letter to the matrix: ")
        matrix_dict[letter] = input_matrix()
    elif action == "2":
        letter = purged_input("Matrix to delete: ")
        if avail_matrix.count(letter) == 1:
            del matrix_dict[letter]
            print("Deleted matrix " + letter)
        else:
            print("Wrong matrix selected")
    elif action == "3":
        letter = purged_input("Input Matrix: ")
        if avail_matrix.count(letter) == 1:
            result = get_cofactor_matrix(matrix_dict[letter])
            if result != False:
                outletter = purged_input("Output Matrix: ")
                matrix_dict[outletter] = result
            else:
                print("Error in matrix")
        else:
            print("Wrong matrix selected")
    elif action == "4":
        letter = purged_input("Input Matrix: ")
        if avail_matrix.count(letter) == 1:
            result = get_adjunct_matrix(matrix_dict[letter])
            if result != False:
                outletter = purged_input("Output Matrix: ")
                matrix_dict[outletter] = result
            else:
                print("Error in matrix")
        else:
            print("Wrong matrix selected")
    elif action == "5":
        letter = purged_input("Input Matrix: ")
        if avail_matrix.count(letter) == 1:
            result = get_inverse_matrix(matrix_dict[letter])
            if result != False:
                outletter = purged_input("Output Matrix: ")
                matrix_dict[outletter] = result
            else:
                print("Error in matrix")
        else:
            print("Wrong matrix selected")
    elif action == "6":
        letter = purged_input("Input Matrix: ")
        if avail_matrix.count(letter) == 1:
            try:
                scalar = Fraction(input("Input Scalar: "))
                result = multiply_matrix_by_scalar(matrix_dict[letter], scalar)
                outletter = purged_input("Output Matrix: ")
                matrix_dict[outletter] = result
            except ValueError:
                print("Invalid Scalar")
        else:
            print("Wrong matrix selected")
    elif action == "7":
        letter = purged_input("Input Matrix 1: ")
        letter2 = purged_input("Input Matrix 2: ")
        if avail_matrix.count(letter) == 1 and avail_matrix.count(letter2) == 1:
            result = matrix_multiplication(matrix_dict[letter],matrix_dict[letter2])
            if result != False:
                outletter = purged_input("Output Matrix: ")
                matrix_dict[outletter] = result
            else:
                print("Error in matrices")
        else:
            print("Wrong matrices selected")
    elif action == "8":
        letter = purged_input("Input Matrix 1: ")
        letter2 = purged_input("Input Matrix 2: ")
        if avail_matrix.count(letter) == 1 and avail_matrix.count(letter2) == 1:
            result = matrix_addition(matrix_dict[letter],matrix_dict[letter2])
            if result != False:
                outletter = purged_input("Output Matrix: ")
                matrix_dict[outletter] = result
            else:
                print("Error in matrices")
        else:
            print("Wrong matrices selected")
    elif action == "9":
        letter = purged_input("Input Matrix: ")
        if avail_matrix.count(letter) == 1:
            print_matrix(matrix_dict[letter])
        else:
            print("Wrong matrix selected")
    elif action == "10":
        letter = purged_input("Input Matrix: ")
        if avail_matrix.count(letter) == 1:
            print("The determinant is " + str(get_determinant(matrix_dict[letter])))
        else:
            print("Wrong matrix selected")
    elif action == "11":
        letter = purged_input("Input Matrix: ")
        if avail_matrix.count(letter) == 1:
            result = get_transposed_matrix(matrix_dict[letter])
            outletter = purged_input("Output Matrix: ")
            matrix_dict[outletter] = result
        else:
            print("Wrong matrix selected")
    elif action == "12":
        letter = purged_input("Input Matrix 1: ")
        letter2 = purged_input("Input Matrix 2: ")
        if avail_matrix.count(letter) == 1 and avail_matrix.count(letter2) == 1:
            compare_matrices(matrix_dict[letter],matrix_dict[letter2])
        else:
            print("Wrong matrices selected")
    elif action == "13":
        letter = purged_input("Matrix to edit: ")
        if avail_matrix.count(letter) == 1:
            matrix_dict[letter] = edit_matrix(matrix_dict[letter])
        else:
            print("Wrong matrix selected")
    elif action == "14":
        letter = purged_input("Output Matrix: ")
        scalar_ok = False
        while scalar_ok == False:
            try:
                min_num = int(input("Enter min. RNG output: "))
                max_num = int(input("Enter max. RNG output: "))
                des_row = int(input("Rows desired: "))
                des_col = int(input("Columns desired: "))
                if min_num <= max_num and des_row > 0 and des_col > 0:
                    scalar_ok = True
                else:
                    if min_num > max_num:
                        print("Min. must be equal or higher than Max.")
                    if des_row <= 0 or des_col <= 0:
                        print("Rows and Columns must be higher than 0")
            except ValueError:
                print("Invalid number")
        matrix_dict[letter] = matrix_rng(des_row, des_col, min_num, max_num)
    elif action == "X":
        want_to_quit = True
    else:
        print("Please enter action again")
