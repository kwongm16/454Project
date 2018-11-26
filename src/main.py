'''
Created on Nov 24, 2018

@author: micha
'''
#Importing Data from Line & Load Excel files
from openpyxl import load_workbook
lineDataFile = load_workbook(filename = 'Line_Data.xlsx')
loadDataFile = load_workbook(filename = 'Load_Data.xlsx')

#Testing printing features
ws = lineDataFile.active
cell_range = ws['A3':'E19'] 
for row in cell_range.values: 
    for value in row: 
        print(value) 
        

print('hello trinh')
