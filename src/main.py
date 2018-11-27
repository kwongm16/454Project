'''
Created on Nov 24, 2018

@author: micha
'''
#Importing data from "Line_Data.xlsx" excel document
from openpyxl import load_workbook
dataFile = load_workbook('Line_Data.xlsx')

#Making each sheet of data (Line & Load data) more accessable
dataFile.sheetnames
lineData = dataFile['Line Data'] 
loadData = dataFile['Load Data'] 

print(lineData.max_row) 
print(loadData.max_row) 

hhiu




