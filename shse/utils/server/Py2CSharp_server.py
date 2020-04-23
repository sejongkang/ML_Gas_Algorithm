from socket import *
from select import *
import sys
from time import ctime, time
import struct
import numpy as np

from shse.data_processing.feature_extraction import *
from shse.data_processing import signal_processing as sp

# wfs Read
from shse.utils.wfsread import wfsread_exp

# import thread module
from _thread import *
import threading

print_lock = threading.Lock()


def test_function(a, b):
    c = a + b
    return c


class test:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        a = test_function(a, b)
        print(a)

    def get_a(self):
        return self.a


class Py2CSharp:
    HOST = ''
    PORT = 7000

    ADDR = (HOST, PORT)

    def recvall(self, sock, n):
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def get_packet(self, result):
        if type(result) == np.ndarray:
            if len(result.shape) == 2:
                format_string_packet = '2;' + str(result.shape[0]) + ';' + str(result.shape[1])
                format_string = str(result.shape[0] * result.shape[1])
                result = result.reshape([-1])
            elif len(result.shape) == 1:
                format_string_packet = '1;' + str(result.shape[0])
                format_string = str(result.shape[0])
            if result.dtype == np.int32:
                format_string_packet += ';i/'
                format_string += 'i'
            elif result.dtype == np.float64 or result.dtype == np.float32:
                format_string_packet += ';d/'
                format_string += 'd'
            result_packet = struct.pack(format_string, *result)
        elif type(result) == str:
            format_string_packet = '1;' + str(len(result)) + ';s/'
            format_string = str(len(result)) + 's'
            result_packet = result.encode('UTF8')
        else:
            if type(result) == np.int32 or type(result) == int:
                format_string_packet = '0;i/'
                format_string = 'i'
            elif type(result) == np.float64 or type(result) == np.float32 or type(result) == float:
                format_string_packet = '0;d/'
                format_string = 'd'
            result_packet = struct.pack(format_string, result)

        format_string_packet = format_string_packet.encode('UTF-8')
        return result_packet, format_string_packet

    def recvdata(self, clientSocket):
        buf_len = self.recvall(clientSocket, 4)
        buf_len = struct.unpack('i', buf_len)
        data = self.recvall(clientSocket, buf_len[0])
        return data

    def senddata(self, clientSocket, data):
        clientSocket.send(struct.pack('i', len(data)))
        clientSocket.send(data)

    def callfunc(self, function_call, parameters):
        result_packet = b''
        format_packet = b''
        results = 0
        print(function_call)
        if function_call.find('=') >= 0:
            exec(function_call)
        else:

            exec('self.results=' + function_call)
            results = self.results
            if type(results) == tuple:
                for i in range(len(results)):
                    result = results[i]
                    packet = self.get_packet(result)
                    result_packet += packet[0]
                    format_packet += packet[1]
            else:
                result = results
                packet = self.get_packet(result)
                result_packet = packet[0]
                format_packet = packet[1]

        return result_packet, format_packet

    def get_fucntion_call(self, function_name, parameters_types, parameters_byte, client_socket):
        function_call = function_name + '('
        offset = 0
        i = 0
        parameters = []
        while offset < len(parameters_byte):
            parameters_format = parameters_types[i].split(';')
            if len(parameters_format) == 2:
                format_string = parameters_format[1]
                parameter = struct.unpack_from(format_string, parameters_byte, offset)
                parameter = parameter[0]
            elif len(parameters_format) == 3:
                if parameters_format[2] == "s":
                    format_string = parameters_format[1] + parameters_format[2]
                    parameter = struct.unpack_from(format_string, parameters_byte, offset)
                    parameter = parameter[0].decode('UTF-8')
                else:
                    format_string = parameters_format[1] + parameters_format[2]
                    parameter = struct.unpack_from(format_string, parameters_byte, offset)
                    parameter = np.array(parameter)
            elif len(parameters_format) == 4:
                format_row = int(parameters_format[1])
                format_col = int(parameters_format[2])
                format_string = str(format_row * format_col) + parameters_format[3]
                parameter = struct.unpack_from(format_string, parameters_byte, offset)
                parameter = np.array(parameter)
                parameter = np.reshape(parameter, newshape=[format_row, -1])

            parameters.append(parameter)
            parameter_name = self.recvdata(client_socket).decode('UTF8')
            if len(parameter_name) > 0:
                parameter_name += '='
            function_call += parameter_name + 'parameters[' + str(i) + '],'
            offset += struct.calcsize(format_string)
            i += 1
        function_call += ')'

        return function_call, parameters

    def run_server(self):
        self.serverSocket = socket(AF_INET, SOCK_STREAM)
        self.serverSocket.bind(self.ADDR)
        self.serverSocket.listen(10)
        connection_list = [self.serverSocket]
        read_socket, write_socket, error_socket = select(connection_list, [], [], 10)

        while True:
            print('[INFO] 요청을 기다립니다...')
            clientSocket, addr_info = self.serverSocket.accept()
            connection_list.append(clientSocket)
            print('[INFO][%s] 클라이언트(%s)가 새롭게 연결 되었습니다.' % (ctime(), addr_info[0]))
            start_new_thread(self.wait_call, (clientSocket,))

    def wait_call(self, clientSocket):
        while True:
            # Recv Data (Call Func)

            function_name = py2charp.recvdata(clientSocket).decode('UTF-8')
            recv_time_start = time()
            parameters_types = py2charp.recvdata(clientSocket).decode('UTF8').split('/')[:-1]
            parameters_byte = py2charp.recvdata(clientSocket)
            recv_time_end = time()
            print('Recv Time: ', (recv_time_end - recv_time_start))


            # Call Function
            call_time_start = time()
            function_call, parameters = py2charp.get_fucntion_call(function_name, parameters_types, parameters_byte, clientSocket)
            result_packet, format_packet = py2charp.callfunc(function_call, parameters)
            call_time_end = time()
            print('Call Time: ', (call_time_end - call_time_start))

            # Send Data
            send_time_start = time()
            py2charp.senddata(clientSocket, format_packet)
            py2charp.senddata(clientSocket, result_packet)
            send_time_end = time()
            print('Send Time: ', (send_time_end - send_time_start))


if __name__ == '__main__':
    py2charp = Py2CSharp()

    try:
        py2charp.run_server()

        # clientSocket, serverSocket = py2charp.run_server()

        # while True:
        #     function_name = py2charp.recvdata(clientSocket).decode('UTF-8')
        #     parameters_types = py2charp.recvdata(clientSocket).decode('UTF8').split('/')[:-1]
        #     parameters_byte = py2charp.recvdata(clientSocket)
        #
        #     function_call, parameters = py2charp.get_fucntion_call(function_name, parameters_types, parameters_byte)
        #     result_packet, format_packet = py2charp.callfunc(function_call, parameters)
        #
        #     py2charp.senddata(clientSocket, format_packet)
        #     py2charp.senddata(clientSocket, result_packet)

    except KeyboardInterrupt:
        # 부드럽게 종료하기
        py2charp.serverSocket.close()
        sys.exit()
