'''
Created on Nov 24, 2018

@author: micha
'''
from openpyxl import load_workbook
wb2 = load_workbook(filename = '454_Data.xlsx')
print(wb2.sheetnames)  


