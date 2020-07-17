#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri 10 July 19:17:15 2020
@author: mcruces

################################
#    Miguel Cruces Fernández   #
#         -------------        #
#    mcsquared.fz@gmail.com    #
#   miguel.cruces@rai.usc.es   #
################################

"""

import os
from os.path import join as joinPath
import sys  # para debug: sys.exit()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages


# # Root directory of the Project
# ROOT_DIR = os.path.abspath("./")
# # Import path to sys.path
# if ROOT_DIR not in sys.path:
#     sys.path.append(ROOT_DIR)
#
# ratePath = joinPath(ROOT_DIR, 'Name,Time,Rates2.txt')
# PathA = joinPath(ROOT_DIR, 'EnvSens/')  # Joao's Data
# PathB = joinPath(ROOT_DIR, 'd_sdg18_posmet_10m.txt')  # Hans' Data #1
# PathC = joinPath(ROOT_DIR, 'Tristan_rates_30062020_Hans.txt')  # Hans' Data #2


class CollectData:
    """
    Class that collects data from the *archive.txt* given
    """

    def __init__(self, rel_path: str,
                 year: int = 0,
                 dt_corr: float = 0,
                 date_fmt: str = 'sex',
                 separator: str = ', ',
                 titles: list = None,
                 time_column_idx: int = 0):

        rootdir = os.path.abspath("./")
        self.txtPath = joinPath(rootdir, rel_path)
        self.year = year
        self.dt_corr = dt_corr
        self.date_fmt = date_fmt
        self.separator = separator
        self.titles = titles
        self.time_column_idx = time_column_idx
        self.time_key = None

        self.data_frame = self.txt2pd()  # Here it's defined self.time_key
        # print(f'Before: {self.time_key}\n', self.data_frame)

        if self.dt_corr:  # FIXME: Añade una línea al RESAMPLEAR
            self.data_frame = self.data_frame.resample('10min', loffset='5min', on=self.time_key).mean().reset_index()
            self.death_time_correction()
        # print(f'After: {self.time_key}\n', self.data_frame)

    def txt2pd(self):
        """
        Data from *archive.txt* is converted to a dictionary.
        The time data is computed to be in datetime format.
        :param date_fmt: Base of the date_ordinal (sexagesimal (sex), decimal (dec)...)
        :param titles: List with titles of the columns. If None it takes the name in the
        *archive.txt*. If the *archive.txt* doesn't have titles, it takes indices from 0
        to N-1 (N columns).
        :param separator:
        :param time_column_idx:
        :return: Python dictionary of arrays with data.
        """
        with open(self.txtPath, 'r') as f:
            file = f.readlines()
            data = []
            for row in file:
                row = row.strip().split(self.separator)
                try:
                    data.append(np.array(row, dtype=np.float))
                except ValueError:
                    self.titles = row
        data = np.asarray(data).transpose().tolist()
        try:
            table = dict(zip(self.titles, data))
        except TypeError:
            table = dict(zip(range(len(data)), data))

        for key in table:
            table[key] = np.array(table[key])
        self.time_key = list(table.keys())[self.time_column_idx]
        table[self.time_key] = self.date_format(table[self.time_key], self.date_fmt)

        return pd.DataFrame(data=table)

    def date_format(self, date_ordinal: object, date_fmt: str):
        """
        It defines the date and time format from the input given.
        :param date_ordinal: Given time input values as array.
        :param date_fmt: Base of the date_ordinal (sexagesimal (sex), decimal (dec)...)
        :return: Array with datetime objects.
        """

        if date_fmt in ['sex', 'SEX']:
            date = []
            date_iter = np.nditer(date_ordinal)
            doy_ = date_iter[0] / 10 ** 6
            for date_time in date_iter:
                doy = int(f'{date_time:0>9f}'[:3])
                date_obj = datetime.fromordinal(doy)
                time = f'{date_time:0>9f}'[3:9]
                date.append(datetime(year=self.year,
                                     month=date_obj.month,
                                     day=date_obj.day,
                                     hour=int(time[:2]),
                                     minute=int(time[2:4]),
                                     second=int(time[4:])
                                     ))  # FIXME: Usar pandas para mejorar esto
                if abs(doy_ - doy) > 2:  # If DOY changes from 365 to 0 => add one year
                    self.year += 1
                doy_ = doy
            return np.asarray(date)
        elif date_fmt in ['doy', 'DOY']:
            date = []
            date_iter = np.nditer(date_ordinal)
            doy_ = date_iter[0]
            for doy in date_iter:
                DOY = int(doy)
                date_obj = datetime.fromordinal(DOY)
                hour = (doy - DOY) * 24
                H = int(hour)
                min_ = (hour - H) * 60
                M = int(min_)
                sec = (min_ - M) * 60
                S = int(sec)
                usec = (sec - S) * 1000
                U = int(usec)
                date.append(datetime(year=self.year,
                                     month=date_obj.month,
                                     day=date_obj.day,
                                     hour=H,
                                     minute=M,
                                     second=S,
                                     # microsecond=U
                                     ))  # FIXME: Usar pandas para mejorar esto
                if abs(doy_ - doy) > 2:
                    self.year += 1
                doy_ = doy
            return np.asarray(date)
        else:
            raise Exception(f"Invalid format: date_fmt = \'{date_fmt}\'. It must be one of them:")

    def death_time_correction_old(self):
        delta = 0.1
        df = self.data_frame.drop(self.time_key, axis=1)  # Delete de time column
        dat_ = df.iloc[0][-1]  # Take the last value in first row to compare with next
        prev_hour = self.data_frame.iloc[0][self.time_key].hour  # Current time (hours)
        for idx, row in df.iterrows():
            dat = row.iloc[-1]  # Last value in next row
            D = abs(dat - dat_) / dat
            curr_hour = self.data_frame.iloc[idx][self.time_key].hour  # Previous time (hours)
            if D >= delta and (curr_hour - prev_hour) >= 3:  # Search error values
                curr_hour = self.data_frame.iloc[idx][self.time_key].hour
                row /= self.dt_corr
                # self.data_frame.set_value(idx, row.keys(), row.values.tolist())
                self.data_frame.at[idx, list(row.keys())] = row.values.tolist()
                dat = row.iloc[-1]
                prev_hour = curr_hour
                # break  # TODO: Ahora solo falta ir metiendo row en self.data_frame
            dat_ = dat  # Store previous value

    def death_time_correction(self):
        prev_hour = self.data_frame.iloc[0][self.time_key].hour  # Current time (hours)
        df = self.data_frame.drop(self.time_key, axis=1)  # Delete de time column
        for idx, row in self.data_frame.iterrows():
            curr_hour = self.data_frame.iloc[idx][self.time_key].hour  # Previous time (hours)
            if row[self.time_key].hour % 3 == 0 and abs(curr_hour - prev_hour) > 1:
                row_ = df.iloc[idx] / self.dt_corr
                self.data_frame.at[idx, list(row_.keys())] = row_
                prev_hour = curr_hour  # Store previous value


if __name__ == '__main__':
    # Generate rePT and repTristan objects
    repPT = CollectData(rel_path='d_sdg18_posmet_10m.txt', year=2018, date_fmt='sex', separator=' ',
                        titles=['Time', 'Lat', 'Lon', 'T', 'P', 'H'])
    repTristan = CollectData(rel_path='Tristan_rates_30062020_Hans.txt', year=2018, date_fmt='doy', separator='\t',
                             dt_corr=0.66)

    # Take data frames from previous objects
    dataPT = repPT.data_frame
    dataTristan = repTristan.data_frame

    # REMOVE DAYS
    # -- Tristan
    # print(f'Before: {repTristan.time_key}\n', dataTristan)
    dataTristan = dataTristan.drop(dataTristan.index[:215]).reset_index(drop=True)  # -215 rows
    # print(f'After: {repTristan.time_key}\n', dataTristan)
    # dataTristan = dataTristan.set_index('Day of Year (dec)')

    # -- PT
    # print(f'Before: {repPT.time_key}\n', dataPT)
    dataPT = dataPT.drop(dataPT.index[2494:]).reset_index(drop=True)  # -1021 rows
    # print(f'After: {repPT.time_key}\n', dataPT)
    # dataPT = dataPT.set_index('Time')

    # RESULT -- COOKING DATA

    result = pd.merge(dataPT, dataTristan, left_index=True, right_index=True)

    # row_del = []  # Rows to delete
    prev = result.iloc[0]
    for idx, row in result.iterrows():
        rate = row['Total Rate[Hz]']
        if rate < 122:
            # row_del.append(idx)
            result.iloc[idx] = prev
        prev = result.iloc[idx]

    result = result.set_index('Day of Year (dec)')
    # result = result.drop(row_del)

    # result = pd.concat([dataTristan, dataPT['T'], dataPT['P']], axis=1, sort=False).drop('index', axis=1)

    # FREQUENCIES -- COOKING DATA

    vrates = np.asarray(result['Total Rate[Hz]'])  # Total rate (Hz)
    vratesMT = np.asarray(result['M1 Total Rate[Hz]'])  # M1 Total Rate (Hz)
    vratesMV = np.asarray(result['M1-Vertical Rate[Hz]'])  # M1-Vertical Rate (Hz)
    vratesMI = np.asarray(result['M1-Inclined[Hz]'])  # M1-Inclined (Hz)

    vrates_n = (vrates - np.nanmean(vrates)) / vrates

    vratesMT_n = (vratesMT - np.nanmean(vratesMT)) / vratesMT

    vratesMV_n = (vratesMV - np.nanmean(vratesMV)) / vratesMV

    vratesMI_n = (vratesMI - np.nanmean(vratesMI)) / vratesMI

    # PRESSURE -- COOKING DATA

    vpress = np.asarray(dataPT['P'])

    # # PLOT PRESSURE
    # plt.figure('Pressure vpress')
    # plt.xlabel('Time')
    # plt.ylabel('Pressure [cPa]')
    # variable = vpress
    # x = np.arange(len(variable))
    # y = variable - np.mean(variable)
    # print('Suma de y media: ', sum(y))
    # plt.plot(x, y, '.')
    # plt.plot(x, np.ones(len(x)) * sum(y))

    # # PLOT FREQUENCY
    # plt.figure('Frequency vrates_n')
    # plt.xlabel('Time')
    # plt.ylabel('Frequency [Hz]')
    # x = np.arange(len(vrates_n))
    # y = vrates_n
    # plt.plot(x, y)

    # set(np.asarray([list(dataPT['Time'].dt.month), list(dataPT['Time'].dt.day)]).T.tolist())

    # TODO: Crear este valor en la clase calculado con una función
    # ndays = len(list(set(list(result['Day of Year (dec)'].dt.date))))
    dates_64 = result.index.values  # All dates
    dates = sorted(list(set(list(pd.to_datetime(dates_64).date))))  # Sorted and not repeated dates
    ndays = len(dates)  # Number of days

    day_ini = 0
    day_fin = ndays  # Cantidad de días que tiene HourlyPressure
    day_int = 2  # Cada cuantos días se cambia de color
    nint = int(ndays / day_int)  # Cuantos cambios de color hay
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0.9, 0.1, nint))  # Desde el azul (90%) hasta el rojo (10%)
    # ---
    # codes: ndays s:single c:centered
    # ---

    # ====================== #
    # ===== COUPLE DAYS ==== #
    # ====================== #

    # ================== PLOT TOTAL ========================
    pmean = 1010  # np.nanmean(vpress)
    icol = 0
    for id in range(nint):  # Iterando sobre la cantidad de colores
        # print('*')
        hini = int(id * day_int * 24 * 6)
        hfim = int(hini + day_int * 24 * 6)
        vpcor = vpress - pmean
        # print(vpcor.shape, vrates_n.shape)
        try:
            # fig = plt.figure(facecolor='white')
            fig = plt.figure(f'Total Rate[Hz] couple day {id}')
            # plt.figure(0)
            plt.title(f'Total Rate[Hz] couple day {id}')
            # plt.xlim(-12.5, 7.5)
            # plt.ylim(4.77, 4.94)
            plt.plot(vpcor[hini:hfim], vrates_n[hini:hfim], 'o', markersize=4,
                     c=colors[icol], label='Rates vs. Pressure', lw=1)
            plt.xlabel('DP = P-Pmean')
            plt.ylabel('DN/N')
            plt.legend(loc='best')
            # plt.show()
            plt.savefig(f'plots/Total Rate[Hz] couple day {id}.png')
            plt.close(fig)
        except ValueError:
            pass
        icol += 1

    # ================== PLOT MT ========================
    icol = 0
    for id in range(nint):  # Iterando sobre la cantidad de colores
        # print('*')
        hini = int(id * day_int * 24 * 6)
        hfim = int(hini + day_int * 24 * 6)
        # print(vpcor.shape, vrates_n.shape)
        try:
            vpcor = vpress[hini:hfim] - np.nanmean(vpress[hini:hfim])
            # fig = plt.figure(facecolor='white')
            fig = plt.figure(f'M1 Total Rate[Hz] couple day {id}')
            plt.title(f'M1 Total Rate[Hz] couple day {id}')
            # plt.xlim(-12.5, 7.5)
            # plt.ylim(4.77, 4.94)
            plt.plot(vpcor, vratesMT_n[hini:hfim], 'o', markersize=4,
                     c=colors[icol], label='Rates vs. Pressure', lw=1)
            plt.xlabel('DP = P-Pmean')
            plt.ylabel('DN/N')
            plt.legend(loc='best')
            # plt.show()
            plt.savefig(f'plots/M1 Total Rate[Hz] couple day {id}.png')
            plt.close(fig)
        except ValueError:
            pass
        icol += 1

    # ================== PLOT MV ========================
    icol = 0
    for id in range(nint):  # Iterando sobre la cantidad de colores
        # print('*')
        hini = int(id * day_int * 24 * 6)
        hfim = int(hini + day_int * 24 * 6)
        # print(vpcor.shape, vrates_n.shape)
        try:
            vpcor = vpress[hini:hfim] - np.nanmean(vpress[hini:hfim])
            # fig = plt.figure(facecolor='white')
            fig = plt.figure(f'M1 Vertical Rate[Hz] couple day {id}')
            plt.title(f'M1 Vertical Rate[Hz] couple day {id}')
            # plt.xlim(-12.5, 7.5)
            # plt.ylim(4.77, 4.94)
            plt.plot(vpcor, vratesMV_n[hini:hfim], 'o', markersize=4,
                     c=colors[icol], label='Rates vs. Pressure', lw=1)
            plt.xlabel('DP = P-Pmean')
            plt.ylabel('DN/N')
            plt.legend(loc='best')
            # plt.show()
            # plt.savefig(f'plots/M1 Vertical Rate[Hz] couple day {id}.png')
            # plt.close(fig)
        except ValueError:
            pass
        icol += 1

    # ================== PLOT MI ========================
    icol = 0
    for id in range(nint):  # Iterando sobre la cantidad de colores
        # print('*')
        hini = int(id * day_int * 24 * 6)
        hfim = int(hini + day_int * 24 * 6)
        # print(vpcor.shape, vrates_n.shape)

        try:
            vpcor = vpress[hini:hfim] - np.nanmean(vpress[hini:hfim])
            # fig = plt.figure(facecolor='white')
            fig = plt.figure(f'M1Inclinated Rate[Hz] couple day {id}')
            plt.title(f'M1 Inclinated Rate[Hz] couple day {id}')
            # plt.xlim(-12.5, 7.5)
            # plt.ylim(4.77, 4.94)
            plt.plot(vpcor, vratesMI_n[hini:hfim], 'o', markersize=4,
                     c=colors[icol], label='Rates vs. Pressure', lw=1)
            plt.xlabel('DP = P-Pmean')
            plt.ylabel('DN/N')
            plt.legend(loc='best')
            # plt.show()
            plt.savefig(f'plots/M1 Inclinated Rate[Hz] couple day {id}.png')
            plt.close(fig)
        except ValueError:
            pass
        icol += 1

    # ====================== #
    # ===== SINGLE DAYS ==== #
    # ====================== #

    day_int = 1  # Cada cuantos días se cambia de color
    nint = int(ndays / day_int)  # Cuantos cambios de color hay
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0.9, 0.1, nint))  # Desde el azul (90%) hasta el rojo (10%)

    # ================== PLOT TOTAL ========================
    pmean = 1010  # np.nanmean(vpress)
    icol = 0
    for id in range(nint):  # Iterando sobre la cantidad de colores
        # print('*')
        hini = int(id * day_int * 24 * 6)
        hfim = int(hini + day_int * 24 * 6)
        vpcor = vpress - pmean
        # print(vpcor.shape, vrates_n.shape)
        try:
            # fig = plt.figure(facecolor='white')
            fig = plt.figure(f'Total Rate[Hz] single day {id}')
            # plt.figure(0)
            plt.title(f'Total Rate[Hz] single day {id}')
            # plt.xlim(-12.5, 7.5)
            # plt.ylim(4.77, 4.94)
            plt.plot(vpcor[hini:hfim], vrates_n[hini:hfim], 'o', markersize=4,
                     c=colors[icol], label='Rates vs. Pressure', lw=1)
            plt.xlabel('DP = P-Pmean')
            plt.ylabel('DN/N')
            plt.legend(loc='best')
            # plt.show()
            plt.savefig(f'plots/Total Rate[Hz] single day {id}.png')
            plt.close(fig)
        except ValueError:
            pass
        icol += 1

    # ================== PLOT MT ========================
    icol = 0
    for id in range(nint):  # Iterando sobre la cantidad de colores
        # print('*')
        hini = int(id * day_int * 24 * 6)
        hfim = int(hini + day_int * 24 * 6)
        # print(vpcor.shape, vrates_n.shape)
        try:
            vpcor = vpress[hini:hfim] - np.nanmean(vpress[hini:hfim])
            # fig = plt.figure(facecolor='white')
            fig = plt.figure(f'M1 Total Rate[Hz] single day {id}')
            plt.title(f'M1 Total Rate[Hz] single day {id}')
            # plt.xlim(-12.5, 7.5)
            # plt.ylim(4.77, 4.94)
            plt.plot(vpcor, vratesMT_n[hini:hfim], 'o', markersize=4,
                     c=colors[icol], label='Rates vs. Pressure', lw=1)
            plt.xlabel('DP = P-Pmean')
            plt.ylabel('DN/N')
            plt.legend(loc='best')
            # plt.show()
            plt.savefig(f'plots/M1 Total Rate[Hz] single day {id}.png')
            plt.close(fig)
        except ValueError:
            pass
        icol += 1

    # ================== PLOT MV ========================
    icol = 0
    for id in range(nint):  # Iterando sobre la cantidad de colores
        # print('*')
        hini = int(id * day_int * 24 * 6)
        hfim = int(hini + day_int * 24 * 6)
        # print(vpcor.shape, vrates_n.shape)
        try:
            vpcor = vpress[hini:hfim] - np.nanmean(vpress[hini:hfim])
            # fig = plt.figure(facecolor='white')
            fig = plt.figure(f'M1 Vertical Rate[Hz] single day {id}')
            plt.title(f'M1 Vertical Rate[Hz] single day {id}')
            # plt.xlim(-12.5, 7.5)
            # plt.ylim(4.77, 4.94)
            plt.plot(vpcor, vratesMV_n[hini:hfim], 'o', markersize=4,
                     c=colors[icol], label='Rates vs. Pressure', lw=1)
            plt.xlabel('DP = P-Pmean')
            plt.ylabel('DN/N')
            plt.legend(loc='best')
            # plt.show()
            plt.savefig(f'plots/M1 Vertical Rate[Hz] single day {id}.png')
            plt.close(fig)
        except ValueError:
            pass
        icol += 1

    # ================== PLOT MI ========================
    icol = 0
    for id in range(nint):  # Iterando sobre la cantidad de colores
        # print('*')
        hini = int(id * day_int * 24 * 6)
        hfim = int(hini + day_int * 24 * 6)
        # print(vpcor.shape, vrates_n.shape)

        try:
            vpcor = vpress[hini:hfim] - np.nanmean(vpress[hini:hfim])
            # fig = plt.figure(facecolor='white')
            fig = plt.figure(f'M1 Inclinated Rate[Hz] single day {id}')
            plt.title(f'M1 Inclinated Rate[Hz] single day {id}')
            # plt.xlim(-12.5, 7.5)
            # plt.ylim(4.77, 4.94)
            plt.plot(vpcor, vratesMI_n[hini:hfim], 'o', markersize=4,
                     c=colors[icol], label='Rates vs. Pressure', lw=1)
            plt.xlabel('DP = P-Pmean')
            plt.ylabel('DN/N')
            plt.legend(loc='best')
            # plt.show()
            plt.savefig(f'plots/M1 Inclinated Rate[Hz] single day {id}.png')
            plt.close(fig)
        except ValueError:
            pass
        icol += 1
