r"""
* 작성자:
    김재영
* 최종 수정날짜:
    2018-10-07.
* 설명:
    타코신호로부터 RPM 계산 함수
"""

import numpy as np

def get_rpm(signal_tacho, fs):
    """
    타코 신호로부터 RPM단위의 회전속도를 계산한다.

    Args:
        signal_tacho: 타코신호(key phaser)
        fs: 샘플링 주파수

    Returns:
        double : RPM(Revolution per minute) 단위의 회전속도

    """
    signal_length = len(signal_tacho)/fs
    min_val = np.min(signal_tacho)
    max_val = np.max(signal_tacho)
    th = (min_val + max_val)/2
    sample_length = len(signal_tacho)
    count=0
    is_over = False
    for i in range(sample_length):
        if signal_tacho[i] >= th and is_over == False:
            count+=1
            is_over = True
        if signal_tacho[i] < th:
            is_over = False

    return (count/signal_length) * 60