import struct

# message type
REQ_DATA_SEND = 1
DATA_SEND = 2
RESPONSE = 3
REQ_DATA_SEND_RES = 4

# response
FAIL = 0
SUCCESS = 1


class Header():
    def __init__(self, buffer):
        # self.struct_fmt = '!4I2BH'    # ubuntu format
        self.struct_format = '=4I2BH'      # window format 수정, 확인
        self.struct_len = struct.calcsize(self.struct_format)

        if buffer != None :
            unpacked = struct.unpack(self.struct_format, buffer)
            self.msg_type = unpacked[0]  # unsigned short (2byte)
            self.body_len = unpacked[1]  # unsigned int   (4byte)
            self.sbc_serial = unpacked[2]   # string (4byte)
            self.daq1_serial = unpacked[3]  # string (4byte)
            self.daq2_serial = unpacked[4]  # string (4byte)

    def get_bytes(self):
        return struct.pack(self.struct_format, *(
            self.msg_type,
            self.body_len,
            self.sbc_serial,
            self.daq1_serial,
            self.daq2_serial
        ))

    def get_size(self):
        return self.struct_len


class BodySBCtoSERVER():    # SBC to SERVER
    def __init__(self, buffer):
        self.struct_fmt = 'HQ10s10s10s' # 수정
        self.struct_len = struct.calcsize(self.struct_fmt)

        if buffer != None:
            unpacked = struct.unpack(self.struct_fmt, buffer)
            self.data_len = unpacked[0]
            self.none = unpacked[1]
        else:
            self.data_len = 0
            self.none = 0

    def Get_bytes(self):
        return struct.pack(self.struct_fmt, *(
                self.data_len,
                self.none
        ))

    def Get_size(self):
        return self.struct_len


class BodySERVERtoSBC():    # SERVER to SBC
    def __init__(self, buffer):
        self.struct_fmt = '2I2d'    # 수정
        self.struct_len = struct.calcsize(self.struct_fmt)

        if buffer != None:
            unpacked = struct.unpack(self.struct_fmt, buffer)
            self.data_len = unpacked[0]
            self.samp_rate = unpacked[1]
            self.samp_time = unpacked[2]
        else:
            self.data_len = 0
            self.samp_rate = 0
            self.samp_time = 0

    def Get_bytes(self):
        return struct.pack(self.struct_fmt, *(
                self.data_len,
                self.samp_rate,
                self.samp_time
        ))

    def Get_size(self):
        return self.struct_len


class BodySendData():   # SBC to SERVER (DATA)
    def __init__(self, buffer):
        if buffer != None:
            self.data = buffer

    def Get_bytes(self):
        return self.data

    def Get_size(self):
        return len(self.data)


class BodyACK():    # SERVER to SBC or SBC to SERVER
    def __init__(self, buffer):
        self.struct_fmt = 'H'
        self.struct_len = struct.calcsize(self.struct_fmt)
        if buffer != None:
            unpacked = struct.unpack(self.struct_fmt, buffer)
            self.response = unpacked[0]
        else:
            self.response = FAIL

    def Get_bytes(self):
        return struct.pack(self.struct_fmt, self.response)

    def Get_size(self):
        return self.struct_len