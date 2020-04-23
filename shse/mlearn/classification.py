import numpy as np
from sklearn import preprocessing
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from shse.data_processing.signal_processing import min_max_normalize

__all__ = ['SVM', 'KNN']


class SVM:
    """ SVM """

    def __init__(self, c=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=1e-3,
                 cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None):
        """

        :param c:
        :param kernel:
        :param degree:
        :param gamma:
        :param coef0:
        :param shrinking:
        :param probability:
        :param tol:
        :param cache_size:
        :param class_weight:
        :param verbose:
        :param max_iter:
        :param decision_function_shape:
        :param random_state:
        """

        self.__rbf_svc = SVC(C=c, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking, probability=probability,
                             tol=tol, cache_size=cache_size, class_weight=class_weight, verbose=verbose, max_iter=max_iter,
                             decision_function_shape=decision_function_shape, random_state=random_state)

    def train(self, x_data, y_data):
        """
        SVM에 해당 데이터 집합을 바탕으로 분류를 수행할 수 있도록 한다 (학습과 유사)

        :param x_data: list, 데이터
        :param y_data: list, 데이터의 결과 [Label]
        :return: None
        """

        # 학습
        # Min-Max Normalize
        self.min = np.min(x_data, 0)
        self.max = np.max(x_data, 0)
        x_data = min_max_normalize(x_data, min_val=self.min, max_val=self.max)

        # Make SVM
        c_range = np.arange(-5, 16, 2, dtype='float')
        c_array = np.power(2, c_range)
        sigma_range = np.arange(-2, 8, 1, dtype='float')
        sigma_array = np.power(2, sigma_range)
        g_array = np.power(np.multiply(np.power(sigma_array, 2), 2), -1)
        rbf_svc = SVC(kernel='rbf', max_iter=-1, probability=True)
        my_scorer = make_scorer(SVM.my_score, needs_proba=True)

        self.__grid_rbf_svc = GridSearchCV(rbf_svc, {'C': c_array, 'gamma': g_array}, scoring=my_scorer)
        self.__grid_rbf_svc.fit(x_data, y_data)

    def classify(self, x_data):
        """
        해당 데이터 분류를 수행함

        :param x_data: list, 분류할 데이터
        :return: list, 분류 결과
        """

        x_data = (x_data - self.min) / (self.max - self.min)
        results = self.__grid_rbf_svc.predict(x_data)
        return results

    def score(self, x_data, y_data):
        """
        해당 데이터를 분류한 후, 그에 따른 정확도를 계산하여 반환한다.

        :param x: list, 데이터
        :param y: list, 데이터의 결과값
        :return: Float, 분류 정확도
        """

        x_data = self.__scaler.transform(x_data)
        return self.__grid_rbf_svc.score(x_data, y_data)

    @staticmethod
    def my_score(y_true, y_pred):
        y_class = np.argmax(y_pred, 1)
        match = y_class + 1 == y_true
        score = np.sum(np.max(y_pred[match, :], 1))

        return score


class KNN:
    """ KNN """

    def __init__(self, n_neighbors=5, weights='distance', metric='euclidean', algorithm='auto',
                 leaf_size=30, p=2, metric_params=None, n_jobs=1, **kwargs):
        """
        KNN의 기본 변수 설정

        :param n_neighbors: Integer, 분류할 경우 판단할 데이터의 수
        :param weights: String, 기본값 = 'distance'
            - 'uniform', 'distance' 존재
        :param metric: String, 기본값 = 'euclidean'
            - 'minkowski', 'euclidean' 존재

        """

        self.__neigh = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric, algorithm=algorithm,
                                            leaf_size=leaf_size, p=p, metric_params=metric_params, n_jobs=n_jobs, **kwargs)

    def train(self, x, y):
        """
        KNN에 해당 데이터 집합을 바탕으로 분류를 수행할 수 있도록 한다 (학습과 유사)

        :param x: list, 데이터
        :param y: list, 데이터의 결과 [Label]
        :return: list, 분류 결과
        """

        self.__neigh.fit(x, y)

    def classify(self, x):
        """
        해당 데이터 분류를 수행함

        :param x: list, 데이터
        :return: list, 분류 결과
        """

        return self.__neigh.predict(x)

    def kneighbors(self, x, n_neighbors, return_distance=True):
        """
        KNN에 삽입한 데이터 집합 중 해당 데이터에서 가장 가까운 데이터들의 인덱스를 반환한다.

        :param x: list, 데이터
        :param n_neighbors: Integer, 찾고자하는 인덱스의 수
        :param return_distance: Bool, 인덱스와 함께 그 인덱스까지의 거리도 반환할지 정하는 변수
        :return: list (인덱스, [거리]), 인덱스와 그에 해당하는 거리 (거리는 return_distance가 True일 경우에만 반환)
        """

        return self.__neigh.kneighbors(x, n_neighbors, return_distance)

    def score(self, x, y):
        """
        해당 데이터를 분류한 후, 그에 따른 정확도를 계산하여 반환한다.

        :param x: list, 데이터
        :param y: list, 데이터의 결과값
        :return: Float, 분류 정확도
        """

        return self.__neigh.score(x, y)


if __name__ == "__main__":
    pass
