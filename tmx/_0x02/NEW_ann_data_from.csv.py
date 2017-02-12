#!/usr/bin/python
# -*- coding: utf-8 -*-
# =========================================================================
import math

# =========================================================================
def normalize( _in0, _in1 ):

    _in = _in0 - _in1;

    if _in > 0:
        res = (1 / (1+math.exp( -_in )))-0.5;

    else:
        res = ((1 / (1+math.exp( _in )))-0.5) * -1;

    r_0 = res;
    r_1 = "{:.3f}".format(res);

    # print( r_0, r_1 );
    return r_1;


# =========================================================================
TIMEFRAME = 1;

FILE_CSV = './new.USD-JPY.'+str(TIMEFRAME)+'m.csv';

_MAX = 50000-1; # 1 == min


_INPUT_LENGTH = 288;
_OUTPUT_LENGTH = 1;

DATA = [];

C = 0;

with open(FILE_CSV, 'r') as FS:
    for line in FS:


        arr = line.split(',');
        # arr[0]; == DATE
        # arr[1]; == TIME
        # A_dO.append( float(arr[2]) );
        # A_dH.append( float(arr[3]) );
        # A_dL.append( float(arr[4]) );
        # A_dC.append( float(arr[5]) );

        DATA.append( float(arr[5]) );
        C += 1;
        if C >= _MAX + _INPUT_LENGTH + 50:
            break;


# print len(DATA); 
# exit();

# =========================================================================
FW_T0 = open('ann.usd-jpy.'+str(TIMEFRAME)+'m.'+str(int(_MAX/1000))+'k.data', "w+");


FW_T0.write( str(_MAX)+" "+str(_INPUT_LENGTH)+" "+str(_OUTPUT_LENGTH)+" \n");

F_STEP = 10;

INDEX = 0;
while INDEX < _MAX:


    row = '';

    i = INDEX;

    while i < (INDEX + _INPUT_LENGTH):


        if( i == INDEX ):
           row += normalize( DATA[i], DATA[i] );
        else:
           row += normalize( DATA[i], DATA[i+1] );

        row += ' ';
        i += 1;

    FW_T0.write( row+"\n");


    f_result = normalize( DATA[i], DATA[i +F_STEP ] );
    FW_T0.write( f_result+" \n");


    print( str(INDEX)+' of '+str(_MAX));
    INDEX += 1;


FW_T0.close();
# =========================================================================
"""

'''
# FRAME
24 * 60 /   5  == 288    # 5  min
24 * 60 /   10 == 144    # 10 min
24 * 60 /   15 == 96     # 15 min
24 * 60 /   30 == 48     # 30 min
24 * 60 /   60 == 24     # 1  H
24 * 60 /  240 == 6      # 4  H
24 * 60 /  720 == 2      # 12 H
24 * 60 / 1440 == 1      # 1  D
'''
FILE_CSV = './new.USD-JPY.1m.csv';
FILE_CSV = './new.USD-JPY.10m.csv';
FILE_CSV = './new.USD-JPY.15m.csv';
FILE_CSV = './new.USD-JPY.30m.csv';
FILE_CSV = './new.USD-JPY.60m.csv';

TIMEFRAME = 1;  # 1440 (cdl) * 1  tfm / 60(min) == 24H
TIMEFRAME = 10; # 144  (cdl) * 10 tfm / 60(min) == 24H
TIMEFRAME = 15; # 96   (cdl) * 15 tfm / 60(min) == 24H
TIMEFRAME = 30; # 48   (cdl) * 30 tfm / 60(min) == 24H
TIMEFRAME = 60; # 24   (cdl) * 60 tfm / 60(min) == 24H

TIMEFRAME = 10; # 144  (cdl) * 10 tfm / 60(min) == 24H

"""
