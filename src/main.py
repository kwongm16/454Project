'''
Created on Nov 24, 2018

@author: micha
'''
#Importing packages used
import openpyxl
import numpy as np
import math
import cmath

#Importing data from "Line_Data.xlsx" excel document
workbook = openpyxl.load_workbook('454 Data.xlsx') 

#Making each sheet of data (Line & Load data) more accessable
lineData = workbook['LineData'] 
busData = workbook['BusData'] 

#Initializing arrays to store P,Q,V,Bus Type
#Where we assume bus 1 is the "Slack Bus" 
P = [] 
Q = [] 
Pgen = [] 
V = [] 
busType = [] 

for row_index in range(1, (busData.max_row+1)): 
    print ("Row:", row_index)
    counter = 1 
    for col_index in range(1, (busData.max_column+1)): 
        if counter == 1: 
            P.append(busData.cell(row=row_index + 1, column=col_index + 1).value)
        elif counter == 2: 
            Q.append(busData.cell(row=row_index + 1, column=col_index + 1).value)
        elif counter == 3: 
            Pgen.append(busData.cell(row=row_index + 1, column=col_index + 1).value)
        elif counter == 4: 
            V.append(busData.cell(row=row_index + 1, column=col_index + 1).value)
        elif counter == 5: 
            busType.append(busData.cell(row=row_index + 1, column=col_index + 1).value)
        else : 
            counter = 1
        
        print ("Column:" , col_index, busData.cell(row=row_index, column=col_index).value)
        counter = counter + 1 
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

for row_index in range(1, (lineData.max_row+1)): 
    print ("Row:", row_index)
    for col_index in range(1, (lineData.max_column+1)): 
        print ("Column:" , col_index, lineData.cell(row=row_index, column=col_index).value)






