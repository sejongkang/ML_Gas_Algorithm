from Message import ISerializable
import struct

class Header(ISerializable):
    def __init__(self, buffer):
        # self.struct_fmt = '!4I2BH'    # ubuntu format
        self.struct_fmt = '=4I2BH'      # window format
        self.struct_len = struct.calcsize(self.struct_fmt)

        if buffer != None :
            unpacked = struct.unpack(self.struct_fmt, buffer)

            self.MSGID = unpacked[0]
            self.MSGTYPE = unpacked[1]
            self.BODYLEN = unpacked[2]
            self.MODE = unpacked[3]
            self.FRAGMENTED = unpacked[4]
            self.LASTMSG = unpacked[5]
            self.SEQ = unpacked[6]

    def Get_bytes(self):
        return struct.pack(self.struct_fmt, *(
            self.MSGID,
            self.MSGTYPE,
            self.BODYLEN,
            self.MODE,
            self.FRAGMENTED,
            self.LASTMSG,
            self.SEQ
        ))

    def Get_size(self):
        return self.struct_len


# MSG_TYPE
REQ_DATA_SEND = 1
DATA_SEND = 2
RESPONSE = 3
REQ_DATA_SEND_RES = 4

# FRAGMENTED
NOT_FRAGMENTED = 0
FRAGMENTED = 1

# LASTMSG
NOT_LASTMSG = 0
LASTMSG = 1

# RESPONSE
FAIL = 0
SUCCESS = 1

class ISerializable:
    def Get_bytes(self):
        pass

    def Get_size(self):
        pass

class Message(ISerializable):
    def __init__(self):
        self.Header = ISerializable()
        self.Body = ISerializable()

    def Get_bytes(self):
        header = self.Header.Get_bytes()
        body = self.Body.Get_bytes()

        return header + body

    def Get_size(self):
        return self.Header.Get_size() + self.Body.Get_size()