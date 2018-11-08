# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 17:02:18 2018

@author: oiluj
"""
from xlrd import open_workbook
import random

values = []
wb = open_workbook('Scaffolds_cromosoma_X.xlsx')

for sheet in wb.sheets():
    number_of_rows = sheet.nrows
    number_of_columns = sheet.ncols

    items = []
    num = 0
    rows = []
    for row in range(3, number_of_rows):
        col = 0
        value  = (sheet.cell(row,col).value)
        values.append(value)


with open('completo.fa') as file, open('/media/juliosullivan/406de5ab-6ba3-4460-958b-da61d0e2372f/ARCHIVOS/equis.fa', 'w') as x, open('/media/juliosullivan/406de5ab-6ba3-4460-958b-da61d0e2372f/ARCHIVOS/noEquis.fa', 'w') as noX:
    for line in file:
        no_es_equis = True
        for id in values:
            if id in line:
                x.write("1\t" + line)
                no_es_equis = False
                break
        if(no_es_equis):
            noX.write("0\t" + line)