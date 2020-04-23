r"""
* 작성자:
    김영훈
* 최종 수정날짜:
    2017-05-27.
* 설명:
    특징추출 함수들
"""

import numpy as np
from numpy.fft import fft


# Peak
def peak(signal):
    """
    신호로부터 Peak를 추출한다.

    Args:
        signal: 입력신호

    Returns:
        double: Peak

    """
    return np.max(np.abs(signal))


# RMS
def rms(signal):
    """
    신호로부터 RMS(Root-Mean-Square)를 추출한다.
    - RMS는 신호의 변화 크기에 대한 값으로 사인 파형과 같이 연속되는 파형의 음과 양을 오가는 정도 또는 그 크기를 의미함

    Args:
        signal: 입력신호

    Returns:
        double: RMS(Root-Mean-Square)

    """
    return np.sqrt(np.sum(np.power(signal, 2)) / len(signal))


# Kurtosiss
def kurtosis(signal, mean=None, std=None):
    """
    신호로부터 Kurtosis를 추출한다.
    - 통계에서 kurtosis는 확률분포의 모양이 뾰족한 정도를 나타내는 지표로 신호의 값들의 분포가 특정 값 근처에 몰려 뾰족한 형태를 이룰수록 kurtosis가 증가함

    Args:
        signal: 입력신호
        mean(optional): 신호의 평균 값
        std(optional): 신호의 표준편차

    Returns:
        double: Kurtosis

    """

    if mean is None: mean = np.mean(signal)
    if std is None: std = np.std(signal, ddof=1)

    return np.sum(np.power((signal - mean) / std, 4)) / len(signal)


# Skewness
def skewness(signal, mean=None, std=None):
    """
    신호로부터 skewness를 추출한다.
    - 통계에서 skewness는 확률 분포의 비대칭성을 나타내는 지표로 신호의 편중성(신호의 평균을 기준으로 신호 값들의 분포가 한쪽으로 몰리는 정도)이 증가할수록 skewness 또한 증가함

    Args:
        signal: 입력신호
        mean(optional): 신호의 평균 값
        std(optional): 신호의 표준편차

    Returns:
        double: Skewness

    """
    if mean is None: mean = np.mean(signal)
    if std is None: std = np.std(signal, ddof=1)

    return np.sum(np.power((signal - mean) / std, 3)) / len(signal)


# square mean root
def smr(signal):
    """
    신호로부터 SMR(Square-Mean-Root)을 추출한다.
    - RMS와 동일하게 연속되는 파형의 음과 양을 오가는 정도 또는 그 크기를 의미하며 RMS보다 신호의 크기에 더 민감함

    Args:
        signal: 입력신호

    Returns:
        double: SMR

    """
    return np.power(np.sum(np.sqrt(np.abs(signal))) / len(signal), 2)


# Peak-to-Peak
def peak2peak(signal):
    """
    신호로부터 Peak-to-Peak를 추출한다.
    - 신호의 전체 폭을 나타내는 지표로 신호에서 가장 작은 값과 가장 큰 값의 차이임

    Args:
        signal: 입력신호

    Returns:
        double: Peak-to-Peak

    """
    return np.max(signal) - np.min(signal)


# Kurtosis factor (Require Kurtosis)
def kurtosis_factor(signal, kurtosis_=None):
    """
    신호로부터 Kurtosis factor를 추출한다.
    - Kurtosis의 변형된 값으로 kurtosis가 신호 전체의 크기에 민감한 단점을 보완한 값임

    Args:
        signal: 입력신호
        kurtosis_(optional): 입력신호의 kurtosis 값

    Returns:
        double: Kurtosis factor

    """

    if kurtosis_ is None: kurtosis_ = kurtosis(signal)

    return kurtosis_ / np.power(np.sum(np.power(signal, 2)) / len(signal), 2)


# Impulse factor
def impulse_factor(signal):
    """
    신호로부터 Impulse factor를 추출한다.
    - 신호에서 가장 큰 임펄스(파형이 뾰족하게 솟아오르는 부분)의 크기에 대한 지표임

    Args:
        signal: 입력신호

    Returns:
        double: Impulse factor

    """
    return np.max(np.abs(signal)) / (np.sum(np.abs(signal)) / len(signal))


# Margin factor (Require SMR)
def margin_factor(signal, smr_=None):
    """
    신호로부터 Margin factor를 추출한다.
    - 신호에서 가장 큰 임펄스(파형이 뾰족하게 솟아오르는 부분)의 크기에 대한 지표임

    Args:
        signal: 입력신호
        smr_(optional): 입력신호의 SMR(Square-Mean-Root)

    Returns:
        double: Margin factor

    """

    if smr_ is None: smr_ = smr(signal)

    return np.max(np.abs(signal)) / smr_


# Crest factor (Require RMS)
def crest_factor(signal, rms_=None):
    """
    신호로부터 Crest factor를 추출한다.
    - Margin과 동일하게 신호의 평균적인 크기에 비해 최소/최대 값의 차이를 의미하며 평균 크기로 SMR대신 RMS를 사용함

    Args:
        signal: 입력신호
        rms_(optional): 입력신호의 RMS(Root-Mean-Square)

    Returns:
        double: Crest factor

    """
    if rms_ is None: rms_ = rms(signal)

    return np.max(np.abs(signal)) / rms_


# Shape factor
def shape_factor(signal, rms_=None):
    """
    신호로부터 Shape factor를 추출한다.
    - 전자공학에서 DC 성분과 AC 성분의 비율을 나타내는 지표로 신호의 평균 대비 음과 양을 오가는 연속 파형의 크기 비율을 의미함

    Args:
        signal: 입력신호
        rms_(optional): 입력신호의 RMS(Root-Mean-Square)

    Returns:
        double: Shape factor

    """
    if rms_ is None: rms_ = rms(signal)

    return rms_ / (np.sum(np.abs(signal)) / len(signal))


def entropy(signal):
    """
    신호로부터 Entropy를 추출한다.
    - 신호의 안정성(Stability)를 나타내는 지표로 entropy가 높을 수록 신호가 불안정함

    Args:
        signal: 입력신호
        rms_(optional): 입력신호의 RMS(Root-Mean-Square)

    Returns:
        double: Entropy

    """
    # hist = np.histogram(signal * 10000, bins=256)  ## When value is too small
    hist = np.histogram(signal, bins=256)
    p = hist[0]
    p = p[p != 0]
    p = p / len(signal)
    return -sum(p * np.log2(p))


def energy(signal):
    """
    신호로부터 Energy를 추출한다.

    Args:
        signal: 입력신호

    Returns:
        double: Energy

    """
    return np.sum(np.power(signal, 2))


def clearance_factor(signal, peak_=None):
    """
    신호로부터 Clearance factor를 추출한다.

    Args:
        signal: 입력신호
        peak_(optional): 입력신호의 peak 값

    Returns:
        double: Clearance factor
    """
    if peak_ is None: peak_ = peak(signal)

    return peak_ / np.power(np.sum(np.sqrt(np.abs(signal))) / len(signal), 2)


def normalize5(signal, mean=None, std=None):
    """
    신호로부터 5차 normalized momentum를 추출한다.

    Args:
        signal: 입력신호
        mean(optional): 입력신호의 평균 값
        std(optional): 입력신호의 표준편차

    Returns:
        double: 5차 normalized momentum
    """
    if mean is None: mean = np.mean(signal)
    if std is None: std = np.std(signal, ddof=1)

    return np.mean(np.power(signal - mean, 5)) / np.power(std, 5)


def normalize6(signal, mean=None, std=None):
    """
    신호로부터 6차 normalized momentum를 추출한다.

    Args:
        signal: 입력신호
        mean(optional): 입력신호의 평균 값
        std(optional): 입력신호의 표준편차

    Returns:
        double: 6차 normalized momentum
    """
    if mean is None: mean = np.mean(signal)
    if std is None: std = np.std(signal, ddof=1)

    return np.mean(np.power(signal - mean, 6)) / np.power(std, 6)


def shape_factor2(signal, peak2peak_=None):
    """
    신호로부터 Shape factor2를 추출한다.

    Args:
        signal: 입력신호
        peak2peak_(optional): 입력신호의 Peak-to-Peak 값

    Returns:
        double: Shape factor2
    """
    if peak2peak_ is None: peak2peak_ = peak2peak(signal)
    return peak2peak_ / (np.sum(np.abs(signal)) / len(signal))


# Frequency Center
def frequency_center(spectrum):
    """
    스펙트럼으로부터 Frequency center를 추출한다.
    - 스펙트럼의 무게중심을 의미함

    Args:
        spectrum: 스펙트럼

    Returns:
        double: Frequency center
    """

    return np.sum(np.arange(1,len(spectrum)) * spectrum) / np.sum(spectrum)


# RMS frequency
def rms_frequency(spectrum):
    """
    스펙트럼으로부터 RMS(Root-Mean-Square)를 추출한다.

    Args:
        spectrum: 스펙트럼

    Returns:
        double: 스펙트럼의 RMS
    """
    return np.sqrt(np.sum(np.power(spectrum, 2)) / len(spectrum))


# Root variance Frequency
def root_variance_frequency(spectrum):
    """
    스펙트럼으로부터 Root variance을 추출한다.

    Args:
        spectrum: 스펙트럼

    Returns:
        double: 스펙트럼의 Root variance
    """
    return np.sqrt(np.sum(np.power(spectrum - np.mean(spectrum), 2)) / len(spectrum))


# Freqeuncy spectrum energy
def frequency_spectrum_energy(spectrum):
    """
    스펙트럼으로부터 Energy를 추출한다.

    Args:
        spectrum: 스펙트럼

    Returns:
        double: 스펙트럼의 Energy
    """
    return np.sum(np.power(spectrum, 2))


def get_all_features(signal, spectrum=None, mean=None, std=None):
    """
    입력신호로부터 모든 특징들을 추출하여 특징벡터로 반환한다.

    Args:
        signal: 입력신호
        spectrum(optional): 입력신호의 스펙트럼
        mean(optional): 입력신호의 평균
        std(optional) 입력신호의 표준편차

    Returns:
        numpy array : 입력신호의 특징 벡터
    """
    
    if mean is None: mean = np.mean(signal)
    if std is None: std = np.std(signal, ddof=1)
    if spectrum is None: spectrum = np.abs(fft(signal))

    # Feature Extraction (Signal)
    features = np.zeros(21)
    features[0] = peak(signal)
    features[1] = rms(signal)
    features[2] = kurtosis(signal, mean=mean, std=std)
    features[3] = crest_factor(signal, rms_=features[1])
    features[4] = impulse_factor(signal)
    features[5] = shape_factor(signal)
    features[6] = skewness(signal, mean=mean, std=std)
    features[7] = smr(signal)
    features[8] = margin_factor(signal, smr_=features[7])
    features[9] = peak2peak(signal)
    features[10] = kurtosis_factor(signal, kurtosis_=features[2])
    features[11] = entropy(signal)
    features[12] = energy(signal)
    features[13] = clearance_factor(signal, peak_=features[0])
    features[14] = normalize5(signal, mean=mean, std=std)
    features[15] = normalize6(signal, mean=mean, std=std)
    features[16] = shape_factor2(signal, peak2peak_=features[9])

    # Feature Extraction (Spectrum)
    features[17] = frequency_center(spectrum)
    features[18] = rms_frequency(spectrum)
    features[19] = root_variance_frequency(spectrum)
    features[20] = frequency_spectrum_energy(spectrum)

    return features


__all__ = ['peak', 'rms', 'kurtosis', 'crest_factor', 'impulse_factor', 'shape_factor', 'skewness', 'smr',
           'margin_factor', 'peak2peak', 'kurtosis_factor', 'entropy', 'energy', 'clearance_factor', 'normalize5',
           'normalize6', 'shape_factor2', 'frequency_center', 'rms_frequency', 'root_variance_frequency',
           'frequency_spectrum_energy', 'get_all_features']

if __name__ == "__main__":
    a = np.array([1, 2, 3, 3, 5, 6, 7])
    b = np.array([1, 1, 2, 2, 1, 1, 1])

    t = {n: v for n, v in zip(b, a)}
    print(t)

    for n, v in zip(b, a):
        print(n, v)
