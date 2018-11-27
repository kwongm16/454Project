'''
Created on Nov 24, 2018

@author: micha
'''
#Importing data from "Line_Data.xlsx" excel document
import openpyxl
import numpy as np
import math
import cmath

workbook = openpyxl.load_workbook('454 Data.xlsx') 

#Making each sheet of data (Line & Load data) more accessable
lineData = workbook['LineData'] 
busData = workbook['BusData'] 

for row_index in range(1, (busData.max_row+1)): 
    print ("Row:", row_index)
    for col_index in range(1, (busData.max_column+1)): 
        print ("Column:" , col_index, busData.cell(row=row_index, column=col_index).value)

print(busData.max_row) 
print(busData.max_column)

print(lineData.max_row) 
print(lineData.max_column)

print(lineData.cell(row = 2, column = 4).value)



