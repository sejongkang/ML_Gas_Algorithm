r"""
* 작성자:
    김영훈
* 최종 수정날짜:
    2017-04-27
* 설명:
    신호처리 함수들
"""

import numpy as np
from scipy.signal import buttord, butter, lfilter
from scipy.signal import hilbert, sosfiltfilt, tf2sos

__all__ = ['spectrum', 'envelope', 'envelope_spectrum', 'low_pass_filter', 'high_pass_filter', 'band_pass_filter']

def spectrum(signal, fs, signal_length) -> (list, list):
    r"""
    신호의 스펙트럼을 추출하여 반환한다.

    Args:
        signal (list): 입력 신호
        fs (float): 샘플링 주파수
        signal_length (int): 신호 길이

    Returns:
        tuple(list, list): Frequency bins, Amplitude

    Examples:
        >>> signal = np.arange(0, 10)  # 신호 데이터
        >>> fs = 25000  # 샘플링 주파수 25kHz
        >>> signal_length = 1  # 신호 길이 1초
        >>> spectrum(signal, fs, signal_length)  # 스펙트럼 추출
        (array([45.        , 44.99999305, 44.99997221, ...,  4.99999432,
                4.99999747,  4.99999937]), array([    0,     1,     2, ..., 12497, 12498, 12499]))

    """

    fft_result = np.fft.fft(signal, fs*signal_length)
    fft_result = np.squeeze(fft_result)
    t_val = np.abs(fft_result)/(fs*signal_length)
    t_val = t_val[0:np.int(len(fft_result) / 2)]

    t_freq = np.arange(0, np.float(fs / 2), np.float(1 / signal_length))

    return t_val, t_freq


def envelope(signal):
    r"""Envelope Function.

    Args:
        signal (list): 입력 신호

    Returns:
        list: 포락 신호

    Examples:
        >>> signal = [0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1]
        >>> envelope(signal)
        array([2.33333333, 2.23606798, 2.60341656, 3.        , 2.60341656,
               2.23606798, 2.33333333, 2.23606798, 2.60341656, 3.        ,
               2.60341656, 2.23606798])
    """

    env_sig = np.abs(hilbert(signal))
    return env_sig


def envelope_spectrum(signal, fs, signal_length):
    """Envelope Spectrum

    Args:
        signal (list): 입력 신호
        fs (float): 샘플링 주파수
        signal_length (int): 신호 길이

    Returns:
        tuple(list, list): 진폭(Amplitude), Frequency bins

    Examples:
        >>> signal = np.arange(0, 10)  # 신호 데이터
        >>> fs = 25000  # 샘플링 주파수 25kHz
        >>> signal_length = 1  # 신호 길이 1초
        >>> envelope_spectrum(signal, fs, signal_length)  # 스펙트럼 추출
        (array([54.37272646, 54.37271237, 54.37267009, ...,  1.34978521,
                1.3494664 ,  1.34927507]), array([    0,     1,     2, ..., 12497, 12498, 12499]))
    """

    env_sig = np.abs(hilbert(signal))
    [tval, tfreq] = spectrum(env_sig, fs, signal_length)
    return tval, tfreq


def low_pass_filter(signal, fs, end_freq):
    """
    Description
     - Low Pass Filter

    Input
     :param signal:list, 입력 신호
     :param fs: Float, 샘플링 주파수 (sampling frequency)
     :param end_freq: Float, 끝 주파수 (passband)

    Output
     :return list, 필터링된 신호

    Example:
        >>> signal = [0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1]
        >>> fs = 0.1
        >>> end_freq = 0.02
        >>> low_pass_filter(signal, fs, end_freq)
        array([-7.04764275e-05,  1.06577490e+00,  2.14355722e+00,  2.64998306e+00,
                2.14375040e+00,  1.06552842e+00, -1.00019323e-03, -1.06409485e+00,
               -2.13795954e+00, -2.66057414e+00, -2.17495858e+00, -9.99904965e-01])
    """

    n, wn = buttord(2 * end_freq / fs, 4 * end_freq / fs, 1, 10)  # 'Butterworth Filter'의 차수(order) 구하기
    b, a = butter(n, wn, btype='low')  # 'Butterworth Filter'의 a와 b 파라메터 생성
    sos = tf2sos(b, a)  # 2차수(order) 필터 계수의 배열 구하기

    signal = np.array(signal).reshape(-1)  # 1차원 리스트가 아닌 경우를 위해 1차원으로 변환

    filtered_signal = sosfiltfilt(sos, signal)  # A forward-backward digital filter using cascaded second-order sections.

    return filtered_signal


def high_pass_filter(signal, fs, start_freq, start_stop_freq):
    """
    Description
     - High Pass Filter

    Input
     :param signal:list, 입력 신호
     :param fs: Float, 샘플링 주파수 (sampling frequency)
     :param start_freq: Float, 시작 주파수 (passband)
     :param start_stop_freq: Float, 시작 주파수의 여유 (stopband)

    Output
     :return list, 필터링된 신호

    Example
    >>> signal = [0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1]
    >>> fs = 0.1
    >>> start_freq = 0.02
    >>> start_stop_freq = 0.03
    >>> high_pass_filter(signal, fs, start_freq, start_stop_freq)
    array([ 0.        ,  0.02923319, -0.12894364,  0.17272625, -0.05544299,
            0.11883306, -0.36247096,  0.09681751,  0.32750824, -0.02172602,
           -0.20884044, -0.34628683])
    """

    n, wn = buttord(start_freq / fs * 2, start_stop_freq / fs * 2, 3, 40)  # 'Butterworth Filter'의 차수(order) 구하기
    b, a = butter(n, wn, btype='high')  # 'Butterworth Filter'의 a와 b 파라메터 생성
    filtered_signal = lfilter(b, a, signal, axis=0)  # A forward-backward digital filter using cascaded second-order sections.

    return filtered_signal


def band_pass_filter(signal, fs, start_freq, end_freq, margin_rate):
    """
    Description
     - High Pass Filter

    Input
     :param signal:list, 입력 신호
     :param fs: Float, 샘플링 주파수 (sampling frequency)
     :param start_freq: Float, 시작 주파수 (passband)
     :param end_freq: Float, 끝 주파수 (passband)
     :param margin_rate: Float, 마진율

    Output
     :return list, 필터링된 신호

    Example
     > signal = [0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1]
     > fs = 0.1
     > start_freq = 0.001
     > end_freq = 0.003
     > band_pass_filter(signal, fs, start_freq, end_freq, 0.08)
       [0.63095463  0.95862525  1.23888854  1.41245392  1.44684504  1.36618891
        1.22107957  1.05946482  0.92849059  0.87631723  0.92292086  1.02981141]
    """

    #
    fn = fs / 2
    wp = np.divide([start_freq, end_freq], fn)
    ws = np.divide([start_freq * margin_rate, end_freq / margin_rate], fn)

    n, wn = buttord(wp, ws, 1, 10)  # 'Butterworth Filter'의 차수(order) 구하기
    b, a = butter(n, wn, btype='band')  # 'Butterworth Filter'의 a와 b 파라메터 생성
    sos = tf2sos(b, a)

    signal = np.array(signal).reshape(-1)
    filtered_signal = sosfiltfilt(sos, signal)

    return filtered_signal


def dwpt(signal, level, mother_func_name):
    # pywt.WaveletPacket(signal, wavelet='db1')
    pass


def min_max_normalize(data, min_val=None, max_val=None):
    min_val = min(data) if min_val is None else min_val
    max_val = max(data) if max_val is None else max_val

    nor_data = (data - min_val) / (max_val - min_val)

    return nor_data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    import pywt

    pass
    file = loadmat(r'E:\Project\SmartHSE\[2017] Backup\changsung\[20170804] feature extraction and SVM\data\[20170621]진동(mat)\기어(30%)\300\ch1\1.mat')
    signal = file['signal']
    # signal_1d = signal.reshape(-1)
    # fs = file['fs'][0][0]
    #
    # print(spectrum(signal))
    # print(pywt.wavelist(), pywt.families())
    #
    # c = pywt.WaveletPacket2D(signal, 'db1', 'shannon')
    #
    # print(c.get_leaf_nodes())
    # plt.plot(c.data)
    # plt.show()
    # print(wv)
    # print(wv2)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax1.plot(signal)
    # signal = [0, 1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1]
    # fs = 0.1
    # start_freq = 0.001
    # end_freq = 0.003
    # a = band_pass_filter(signal, fs, start_freq, end_freq, 0.08)
    # print(a)
    #
    # filtered_signal = low_pass_filter(signal, 65536, 5000, 6000)
    # x = np.reshape(filtered_signal, -1)
    # ax2 = fig.add_subplot(212)
    # ax2.plot(x)
    # plt.show()
