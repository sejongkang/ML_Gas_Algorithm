from shse.network.socket.packet import *


class ISerializable():
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


class MessageUtil():
    @staticmethod
    def Header_set(mode, msg, msgid, msgtype, bodylen, fragment, lastmsg, seq):
        msg.Header = Header(None)
        msg.Header.MODE = mode
        msg.Header.MSGID = msgid
        msg.Header.MSGTYPE = msgtype
        msg.Header.BODYLEN = bodylen
        msg.Header.FRAGMENTED = fragment
        msg.Header.LASTMSG = lastmsg
        msg.Header.SEQ = seq

    @staticmethod
    def send(sock, msg):
        sent = 0
        buffer = msg.Get_bytes()
        while sent < msg.Get_size():
            sent += sock.send(buffer)

    @staticmethod
    def receive(sock):
        totalRecv = 0
        sizeToRead = 20
        hbuffer = bytes()

        while sizeToRead > 0 :
            buffer = sock.recv(sizeToRead)
            if not buffer:
                return None

            hbuffer += buffer
            totalRecv += len(buffer)
            sizeToRead -= len(buffer)

        if totalRecv != 20:
            return None

        header = Header(hbuffer)

        totalRecv = 0
        bbuffer = bytes()
        sizeToRead = header.BODYLEN

        while sizeToRead > 0:
            buffer = sock.recv(sizeToRead)
            if not buffer:
                return None

            bbuffer += buffer
            totalRecv += len(buffer)
            sizeToRead -= len(buffer)

        if totalRecv != header.BODYLEN:
            return None

        body = None

        if header.MSGTYPE == Message.REQ_DATA_SEND:
            body = BodyRequestDataSend(bbuffer)
        elif header.MSGTYPE == Message.RESPONSE:
            body = BodyResponse(bbuffer)
        elif header.MSGTYPE == Message.DATA_SEND:
            body = BodyData(bbuffer)
        elif header.MSGTYPE == Message.REQ_DATA_SEND_RES:
            body = BodyRequestDataSendResponse(bbuffer)
        else:
            raise Exception(
                "Unknown MSGTYPE : {0}".format(header.MSGTYPE))

        msg = Message.Message()
        msg.Header = header
        msg.Body = body

        return msg