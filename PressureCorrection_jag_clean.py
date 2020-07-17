#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
#############
@JPCS_04/2020
joao.saraiva@coimbra.lip.pt
#############
"""
import os
from os.path import join as joinPath
import sys  # para debug: sys.exit()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib.backends.backend_pdf import PdfPages

# Directorio raíz del proyecto
ROOT_DIR = os.path.abspath("./")
# Se importa a sys.path
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# fullPath = "/Users/JAG/progs/python/tristan/joao/"
fullPath = ROOT_DIR
fullPath2 = ROOT_DIR
ratePath = joinPath(fullPath, 'Name,Time,Rates2.txt')
pressurePath = joinPath(fullPath, 'EnvSens/')  # Datos de Joao
# fullPath2 = "/Users/JAG/progs/python/tristan/data10m/"
pressurePathx = joinPath(fullPath2, 'd_sdg18_posmet_10m.txt')  # Datos de Hans
# save figures
'''
2018 Nov16: DOY = 320 
2018 Dic12: DOY = 346
'''

########################################
# Read And Customise Data Frame (Joao) #
########################################

# Passar os 754 valores de correctedRates para uma média horária; e assim obter apenas 672 valores (28dias x 24h)

df = pd.read_csv(ratePath, sep='\t', header=None, engine='c')  # Data Frame
print('DF Before: \n', df.iloc[0:5])
xdf = df
df['%Y-%m-%d %H:%M:%S'] = pd.to_datetime(df[0], format='%Y-%m-%dT%H%M%S')
df = df.set_index('%Y-%m-%d %H:%M:%S').resample('1H').mean()
ydf = df
print('DF After: \n', df.iloc[0:5])
dfh = df  # datos corregidos en horas

HourlyRates_16Nov13Dez = list(df[2])
print('* HourlyRates ', len(HourlyRates_16Nov13Dez))
# 672 valores de cosmic rates corrigidas (random+eff), de 16Nov2018_0h a 13Dez_23h (28dias x 24h; NaN se nao há valor)
HourlyRates_16Nov13Dez_NaperianLog = np.log(HourlyRates_16Nov13Dez)
i16nov12 = 12
i12dec12 = -36
vr = HourlyRates_16Nov13Dez[i16nov12:i12dec12]
vrl = HourlyRates_16Nov13Dez_NaperianLog[i16nov12:i12dec12]

########################################
# Read And Customise Data Frame (Hans) #
########################################

df2 = pd.read_csv(pressurePathx, sep=' ')  # , sep='\t', header=None, engine = 'c')

df2 = df2.set_axis(['Time', 'Lat', 'Lon', 'T', 'P', 'H'], axis=1, inplace=False)

v2p = list(df2['P'])  # List with all Pressures
v2t = list(df2['Time'])  # List with all Times

# Precisamos das datas respetivas, numa lista que possa ser usada como eixo dos XX no plot final ao comparar taxas
# corrected vs. nao corrected.
# Formas de converter o index em lista de timestamps: df.index.tolist(), list(df.index), df.index.values.tolist()
listTimestamps = [int(item / 1000000000) for item in df.index.values.tolist()]
#
from datetime import datetime

months = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun', 7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out',
          11: 'Nov', 12: 'Dez'}

DatajaVista = ''
dataFinal = []  # lista de datas para usar no eixo dos XX do último plot de taxas corrected vs. nao corrected
dates = [datetime.fromtimestamp(timestamp) for timestamp in listTimestamps]
for date in dates:
    month = months[date.month]
    # if date.month == 11:
    #     month = 'Nov'
    # elif date.month == 12:
    #     month = 'Dez'
    # else:
    #     month = 'month'  # Pra evitar erros, em casso de nao ficar definido
    if date.day == DatajaVista:  # Como usamos em baixo ticker.MultipleLocator(base=24), basta ter a lista das datas
        # dataFinal.append('')      #dataFinal nao precisa de ter 672 valores, bastam 28, as 28 datas
        pass
    else:
        # dataFinal.append(str(date.day) + ' ' + month)
        dataFinal.append(f'{date.day:0>2d} {month}')
        DatajaVista = date.day
print('* Len(dataFinal) ', len(dataFinal))
# print(dataFinal)

# Passar tb os valores de pressao para médias horárias; tb 672 valores (28dias x 24h)
from os import listdir
from os.path import isfile

listOfFiles = [f for f in listdir(pressurePath) if isfile(joinPath(pressurePath, f))]
# print(listOfFiles)
listOfFiles.sort()
HourlyPressure = []
for File in listOfFiles:
    df = pd.read_csv(joinPath(fullPath, 'EnvSens/' + File), sep=' ', header=None, engine='c')
    # print(df.iloc[0:3])
    df['%Y-%m-%d %H:%M:%S'] = pd.to_datetime(df[0])
    hour = pd.to_timedelta(df['%Y-%m-%d %H:%M:%S'].dt.hour, unit='H')
    hour.name = 'Hour'
    result = df.groupby(hour).mean()
    HourlyPressure.extend(list(result[7] * 10))  # passar os valores para hPa em vez de kPa
ldata = len(HourlyPressure)
print('* Len(HourlyPressure) ', ldata)
# 672 (28 days) valores de pressao; média horária a partir de 16Nov2018 0h

# com base na sugestao do Junjo, usa-se para Po a média entre o 1º e ultimo valore de pressao
averagedPressure = (HourlyPressure[-1] + HourlyPressure[0]) / 2
print('* Aver. Pressure ', averagedPressure)  # Po
HourlyPressureMinusAvPressure = [pressureValue - averagedPressure for pressureValue in HourlyPressure]

vrates = HourlyRates_16Nov13Dez
vrates_nlog = HourlyRates_16Nov13Dez_NaperianLog
visnan = np.isnan(vrates)
vpress = HourlyPressure

ndays = int(ldata / 24)  # Cantidad de horas totales que tiene HourlyPressure / 24h
day_ini = 0
day_fin = ndays  # Cantidad de días que tiene HourlyPressure
day_int = 2  # Cada cuantos días se cambia de color
nint = int(ndays / day_int)  # Cuantos cambios de color hay
cmap = plt.get_cmap('jet')
colors = cmap(np.linspace(0.9, 0.1, nint))  # Desde el azul (90%) hasta el rojo (10%)
# ---
fbout = fullPath2 + 'bild_prescor_4d.pdf'
# codes: ndays s:single c:centered
# ---
pmean = np.mean(vpress)

with PdfPages(fbout) as pdf:
    icol = 0
    for id in range(nint):  # Iterando sobre la cantidad de colores
        # print('*')
        hini = int(id * day_int * 24)
        hfim = int(hini + day_int * 24)
        vpcor = vpress - pmean

        # fig = plt.figure(facecolor='white')
        plt.figure(1)
        plt.xlim(-10., 10.)
        plt.ylim(5.0, 5.175)
        plt.plot(vpcor[hini:hfim], vrates_nlog[hini:hfim], 'o', markersize=4,
                 c=colors[icol], label='loge_Rates vs. Pressure', lw=1)
        # plt.legend(loc='best')
        icol += 1
        plt.show()
        # pdf.savefig()
