r"""
* 작성자:
    김재영
* 최종 수정날짜:
    2018-10-07.
* 설명:
    Orbit을 그리기 위한 좌표데이터 계산 함수
"""

from shse.data_processing.get_rpm import *
import numpy as np
from scipy import signal
from numpy import sin, cos
from struct import unpack
import scipy.io as sio

def orbit(signal_x, signal_y, signal_tacho, fs, rot):
    """
    수평, 수직 방향의 변위신호와 타코신호를 통해 축의 궤도에 대한 x,y 좌표들과 high spot의 x,y좌표를 반환한다.

    Args:
        signal_x: 수평(x 축) 변위신호
        signal_y: 수직(y 축) 변위신호
        signal_tacho: 타코신호(key phaser)
        fs: 샘플링 주파수
        rot: 수평으로부터 센서의 각도
        
    Returns:
        tuple(double, double, double, double): Orbit의 x축 좌표들, Orbit의 y축 좌표들, High spot의 x 좌표, High spot의 y 좌표

    Examples:
        >>> signal_x, signal_y, signal_tacho # x, y 변위신호와 tacho 신호 (main 소스 참조)
        >>> fs = 25000  # 샘플링 주파수 25kHz
        >>> rot=45 # 수평축으로부터 센서의 각도
        >>> x, y, hx, hy=orbit(signal_x, signal_y, signal_tacho, fs, rot)

    """
    x = signal_x
    y = signal_y

    rpm = get_rpm(signal_tacho, fs)

    a = np.where(abs(signal_tacho)>max(abs(signal_tacho))*3/5)[0]
    hs = np.where(np.diff(a)!=1)[0]
    b2, a2 = signal.butter(6, 5*(rpm/60)/(fs/2), 'low')
    dataInX = x
    x_data = signal.lfilter(b2,a2,dataInX)
    dataInY = y
    y_data = signal.lfilter(b2,a2,dataInY)

    b1, a1 = signal.butter(1,[((rpm-rpm*0.1)/60)/(fs/2), ((rpm+rpm*0.1)/60)/(fs/2)],'bandpass')
    dataInX = x_data
    x_data = signal.lfilter(b1, a1, dataInX)
    dataInY = y_data
    y_data = signal.lfilter(b1, a1, dataInY)
    rad = rot*np.pi/180
    RCDis = np.matmul([[cos(rad), -sin(rad)],[sin(rad), cos(rad)]], [x_data, y_data])

    x_data = RCDis[0]
    y_data = RCDis[1]

    hx = x_data[a[hs]]
    hy = y_data[a[hs]]

    return x_data, y_data, hx, hy

if __name__ == '__main__' :
    folder_path = 'C:/Users/JY/Desktop/[2018] 연구실/[2018] 한전 KPS 시연/bode_data'
    x_file_path = folder_path + '/x.txt'
    y_file_path = folder_path + '/y.txt'
    tacho_file_path = folder_path +'/tacho.txt'

    f = open(x_file_path,'rb')
    fs = int(unpack('d',f.read(8))[0])
    slen = unpack('d',f.read(8))[0]
    signal_x = np.asarray(unpack(str(int(fs*slen))+'d', f.read(int(fs*slen*8))))
    f.close()

    f = open(y_file_path, 'rb')
    fs = int(unpack('d', f.read(8))[0])
    slen = unpack('d', f.read(8))[0]
    signal_y = np.asarray(unpack(str(int(fs*slen))+'d', f.read(int(fs*slen*8))))
    f.close()

    f = open(tacho_file_path, 'rb')
    fs = int(unpack('d', f.read(8))[0])
    slen = unpack('d', f.read(8))[0]
    signal_tacho = np.asarray(unpack(str(int(fs*slen))+'d', f.read(int(fs*slen*8))))
    f.close()

    x, y, hx, hy=orbit(signal_x[fs*170:fs*171], signal_y[fs*170:fs*171], signal_tacho[fs*170:fs*171], fs, 45)

    sio.savemat('orbit.mat',{'x':x, 'y':y, 'hx':hx, 'hy':hy})







