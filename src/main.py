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
workbook.sheetnames
lineData = workbook['Line Data'] 
loadData = workbook['Load Data'] 

print(lineData.max_row) 
print(loadData.max_row) 




