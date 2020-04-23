"""
Author : Younghun Kim111\n
Modified Date : 2017-04-27\n
Description : Neural Network using Pytorch\n
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from shse.mlearn.utils import *
from torch.utils import data as utils_data

__all__ = ['DNN', 'CNN']

MAX_POOL = -1


class _BaseLayer(nn.Module):
    """ 모든 뉴럴 네트워크의 부모 클래스 """

    def __init__(self, activation_func, previous_model=None, **kwargs):
        """
        Description
        - NeuralNet Class 생성자 함수, 'Neural Network'에 필요한 변수들을 초기화함

        Input
        :param activation_func: function (torch.nn.modules.activation), 활성함수를 나타냄
        :param previous_model: object, 현재 모델 앞에 연결할 다른 모델

        Output
        :return None

        Example
         > neural_1 = NeuralNet(nn.ReLU)
         > neural_2 = NeuralNet(nn.ReLU, previous_model=(neural_1,))

         위의 예시를 수행할 경우 다음과 같은 네트워크 구조를 가짐
           input --- neural_1 ---- neural_2 --- output

        """

        # Call Parent Class(nn.Module)
        super().__init__()

        # Init Variables
        self.layers = nn.ModuleList()
        self.activation_func = activation_func()
        self.previous_model = previous_model or []

        # Check GPU
        self.gpu = torch.cuda.is_available()  # Check Support CUDA
        for device_number in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(device_number)

            if (device.major == 3 and device.minor == 0) or device.major < 3:
                self.gpu = False

            # name = torchget_device_name(d)
            # if CUDA_VERSION < 8000 and major >= 6:
            #     warnings.warn(incorrect_binary_warn % (d, name, 8000, CUDA_VERSION))
            # elif CUDA_VERSION < 9000 and major >= 7:
            #     warnings.warn(incorrect_binary_warn % (d, name, 9000, CUDA_VERSION))
            # elif capability == (3, 0) or major < 3:
            #     warnings.warn(old_gpu_warn % (d, name, major, capability[1]))

    def forward(self, x):
        """
        Neural Network 모델의 (예측)결과를 반환

        :param x: Variable (Torch.autograd.Variable), 데이터 변수

        :return: Variable (Torch.autograd.Variable), 모델 결과

        :Example
          > neural_1 = DNN(4, 4, hidden_layer=(), nn.ReLU, previous_model=None)
          > neural_1
            DNN(
              (layers): Sequential(
                (layer_1): Linear(in_features=4, out_features=4, bias=True)
              )
              (activation_func): ReLU()
            )

          > data = torch.FloatTensor(3, 4)
          > data
            [[0.1, 0.22, 1, 0.1],
             [0.1, 0.2, 0.1, 0.1],
             [0.22, 0.3, 0.5, 0.7]]

          > data_var = Variable(data)
          > neural_1.forward(data_var)
            Variable containing:
             0.8568 -0.2065 -0.0134 -0.4088
             0.6099 -0.4145  0.1082 -0.2655
             1.0484 -0.6188 -0.2065 -0.5642
            [torch.FloatTensor of size 3x4]

        """

        # Check data set
        if not torch.is_tensor(x):  # Check Data
            x = torch.from_numpy(x)

        prev_x = []
        for prev_mode in self.previous_model:
            prev_x = np.hstack((prev_x, prev_mode.forward(x)))
            # x = self.activation_func(x)

        if prev_x:
            x = self.activation_func(prev_x)

        return x

        # for index, _ in enumerate(self.layers):
        # return self.layers(x)

    def learn(self, data, labels, epoch, learning_rate=1e-3, loss_func=nn.CrossEntropyLoss, batch_size=None):
        """

        :param data: Tensor
        :param labels:
        :param epoch:
        :param learning_rate:
        :param loss_func:
        :param batch_size:
        """
        # Check GPU and Change Mode (GPU, CPU)
        self._set_gpu()

        # Check data set
        if not torch.is_tensor(data):  # Check Data
            data = torch.from_numpy(data)
        if not torch.is_tensor(labels):  # Check Labels
            labels = torch.from_numpy(labels)

        # Set Batch Size
        if batch_size is None or batch_size > len(data):
            batch_size = len(data)

        # Set DataLoader
        data_set = utils_data.TensorDataset(data, labels)
        data_loader = utils_data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

        # Set Optimizer
        loss_function = loss_func()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Training
        loss = -1
        for step in range(epoch):  # Epoch
            for index, (data, labels) in enumerate(data_loader):  # Batch
                # Set Data(X), Labels(Y)
                if self.gpu:
                    data = data.cuda()
                    labels = labels.cuda()

                # Training
                optimizer.zero_grad()  # Init Gradient Value
                output = self.forward(data)  # Forward
                loss = loss_function(output, labels)  # Get Loss
                loss.backward()  # Back Propagation
                optimizer.step()  # Update Weight, Bias

            yield step, loss

    def test(self, data):
        # Check GPU and Change Mode (GPU, CPU)
        self._set_gpu()
        self.eval()  # dd
        if not torch.is_tensor(data):
            data = torch.from_numpy(data)

        if self.gpu:
            data = data.cuda()

        output = self.forward(data)
        return output

    def save(self, file_name):
        save_model(self, file_name)

    def _set_gpu(self):
        # Check GPU and Change Mode (GPU, CPU)
        if self.gpu:
            self.cuda()
        else:
            self.cpu()

    def add_previous_mode(self, model):
        # self.previous_model.append(model)
        self.previous_model = model

    def set_train_model(self, mode=None):
        super().train(mode)


class FullyConnectedLayer(_BaseLayer):
    def __init__(self, input_size, output_size, hidden_layer=(), activation_func=nn.ReLU, softmax=False,
                 norm_func=nn.BatchNorm1d, previous_model=None, **kwargs):
        super().__init__(activation_func=activation_func, previous_model=previous_model, **kwargs)

        self.fc_layer = nn.Sequential()

        # Make Layers
        layer_array = np.hstack((input_size, hidden_layer, output_size))  # Set Layer Array
        for index, _ in enumerate(layer_array[:-1]):
            layer = nn.Linear(int(layer_array[index]), int(layer_array[index + 1]))
            self.fc_layer.add_module('layer_' + str(index + 1), layer)

            # 마지막은 제외하고 Activation Function 추가
            if index < len(layer_array) - 2:
                self.fc_layer.add_module('normalization_' + str(index + 1), norm_func(int(layer_array[index + 1])))
                self.fc_layer.add_module('activation_' + str(index + 1), activation_func())
            # 마지막(아웃풋)은 Softmax Activation Function 추가
            elif softmax:
                self.fc_layer.add_module('softmax' + str(index + 1), nn.Softmax())

        # self.layers.append(fc_layer)
        self.layers.add_module('FC_Layer_' + str(len(self.layers)), self.fc_layer)

    def forward(self, x):
        out = super().forward(x)
        out = out.view(out.shape[0], -1)
        out = self.fc_layer(out)
        return out


class DNN(FullyConnectedLayer):
    """
    DNN (Deep Neural Network)
    """
    def __init__(self, input_size, output_size, hidden_layer=(), activation_func=nn.ReLU, softmax=False,
                 norm_func=nn.BatchNorm1d, previous_model=None, **kwargs):
        """
        DNN의 초기화 함수

        :param input_size:
        :param output_size:
        :param hidden_layer:
        :param activation_func:
        :param softmax: LossFunction가 CrossEntropy일 때는 False
        :param norm_func:
        :param previous_model:
        :param kwargs:
        """

        super().__init__(input_size=input_size, output_size=output_size, hidden_layer=hidden_layer, activation_func=activation_func, softmax=softmax,
                         norm_func=norm_func, previous_model=previous_model, **kwargs)

    pass


class ConvolutionLayer(_BaseLayer):
    # def __init__(self, channels=(), kernel_size=3, stride=1, padding=1, activation_func=nn.ReLU,  dimension=1, convolution_func=nn.Conv1d,
    #              norm_func=nn.BatchNorm1d, pooling_func=nn.MaxPool1d, pooling_size=2, pooling_stride=2, previous_model=None, **kwargs):
    def __init__(self, channels=(), kernel_size=3, stride=1, padding=1, dimension=1, activation_func=nn.ReLU,
                 pooling_size=2, pooling_stride=2, previous_model=None, **kwargs):

        super().__init__(activation_func=activation_func, previous_model=previous_model, **kwargs)

        self.conv_layer = nn.Sequential()

        if dimension == 1:
            convolution_func = nn.Conv1d
            norm_func = nn.BatchNorm1d
            pooling_func = nn.MaxPool1d
        elif dimension == 2:
            convolution_func = nn.Conv2d
            norm_func = nn.BatchNorm2d
            pooling_func = nn.MaxPool2d
        else:
            ''' DIMENSION ERROR '''
            logging.error('ValueError: dimension must be 0 or 1, but got {}'.format(dimension))
            return
            # logging.error('File "E:/Project/SmartHSE/shse_library/shse/mlearn/neural_network.py", line 265, in <module> [4, 5, 5]])')

        # Make Layers
        conv_channels, pooling_index = ConvolutionLayer.split_channel_pool(channels)
        for index, _ in enumerate(conv_channels[:-1]):
            conv = convolution_func(int(conv_channels[index]), int(conv_channels[index + 1]),
                                    kernel_size=kernel_size, stride=stride, padding=padding)
            self.conv_layer.add_module('convolution_' + str(index + 1), conv)

            # Normalization and Activation
            self.conv_layer.add_module('normalization_' + str(index + 1), norm_func(int(conv_channels[index + 1])))
            self.conv_layer.add_module('activation_' + str(index + 1), activation_func())

            # Pooling Layer
            if index in pooling_index:
                maxpool = pooling_func(pooling_size, pooling_stride)
                self.conv_layer.add_module('pooling_' + str(index + 1), maxpool)

        # self.layers.append(conv_layer)
        self.layers.add_module('Conv_Layer_' + str(len(self.layers)), self.conv_layer)

    @staticmethod
    def split_channel_pool(channels):
        channels = np.array(channels)
        pooling_index = [pooling_index - 2 - index for index, pooling_index in
                         enumerate(np.where(channels == MAX_POOL)[0])]  # 왜지 왜 numpy만 되지
        channels = channels[channels != MAX_POOL]

        return channels, pooling_index

    def forward(self, x):
        """
        CNN의 Forward 수행

        :param x:
        :return:
        """
        out = super().forward(x)
        out = self.conv_layer(out)
        return out


# class CNN(FullyConnectedLayer, ConvolutionLayer):
class CNN(FullyConnectedLayer, ConvolutionLayer):
    def __init__(self, input_size, output_size, channels=(), fc_layers=(), kernel_size=3, stride=1, padding=1, dimension=1,
                 activation_func=nn.ReLU, pooling_size=2, pooling_stride=2, previous_model=None, **kwargs):

        conv_channels, pooling_index = self.split_channel_pool(channels)
        for i, _ in enumerate(conv_channels[:-1]):
            input_size = np.floor(((input_size - kernel_size + (2 * padding)) / stride) + 1)

            if i in pooling_index:
                input_size = np.floor(((input_size - pooling_size) / pooling_stride) + 1)

        input_size = input_size * conv_channels[-1]

        super().__init__(input_size=input_size, output_size=output_size, channels=channels, hidden_layer=fc_layers,
                         kernel_size=kernel_size, stride=stride, padding=padding, dimension=dimension,
                         activation_func=activation_func, pooling_size=pooling_size, pooling_stride=pooling_stride,
                         previous_model=previous_model, **kwargs)


class RNN(_BaseLayer):
    def __init__(self, input_size, output_size, hidden_size, activation_func=nn.ReLU, previous_model=None):
        super().__init__(activation_func)


if __name__ == '__main__':
    # dnn = DNN(3, 4, [4, 4])
    # cnn = CNN(4, 2, [1, 20, -1, 10, 10, -1, 20, 30, 50, -1, 20, 30], dimension=1)
    cnn = CNN(10, 2, [1, 20, 30, -1, 10], [10, 20], dimension=1, kernel_size=3, stride=3, padding=22)

    data = torch.Tensor([[[1., 2., 3., 4, 5, 6, 7, 8, 9, 10]],
                         [[2., 3., 3., 4, 5, 6, 1, 2, 3, 4]],
                         [[4., 5., 6., 7, 8, 9, 7, 8, 9, 10]]])
    print(data.shape)
    # data_dnn = torch.Tensor([[1., 2., 3.],
    #                          [2., 3., 4.],
    #                          [3., 4., 5.]])
    #
    # y = cnn.forward(data)
    # y2 = cnn.layers

    # print(cnn)
    # print(cnn.layers)
    #
    # cnn.forward([[1., 2., 3.]])
    # print(dnn.forward(data_dnn))
    # print(y)
    # print(y.shape)
    # print(dnn.forward(data_dnn))
    #
    label = torch.Tensor([0, 1, 1]).long()
    # #
    for i in cnn.learn(data, label, epoch=1, learning_rate=0.05):
        if i[0] % 10 == 0:
            print(i)

    result = cnn.test(torch.unsqueeze(data[0], dim=0)).tolist()
    print(result)
    print(np.argmax(result))

    # save_model(cnn, 'file.pkl')
    # d = load_model()
    # print('dnn')
    # for i in dnn.learn(data_dnn, label, epoch=10, learning_rate=0.05):
    #     pr