import numpy as np
import struct
from collections import namedtuple
import math

def read_packets(f, rsize, skip_size):
    if rsize == -1:
        bytes = b''
        kk=1
        while 1:
            try:
                raw_buf = f.read(4096 * 2)
                bytes += raw_buf
                f.seek(skip_size, 1)
            except:
                break
        raw_buf = f.read()
        bytes += raw_buf
    else:
        if rsize < 4096:
            bytes = f.read(rsize * 2)
        else:
            bytes = b''
            for kk in range(1, rsize - 4096, 4096):
                raw_buf = f.read(4096 * 2)
                bytes += raw_buf
                f.seek(skip_size, 1)
            raw_buf = f.read(rsize % 4096 * 2)
            bytes += raw_buf
    channel = np.asarray(struct.unpack(str(int(len(bytes) / 2)) + 'h', bytes))
    read_size = len(channel)
    return channel, read_size

def wfsread_exp(filename, start_time=None, end_time=None, start_chnum=None, end_chnum=None):
    Number_of_channels,Sample_rate,Max_voltage,Header_length, delay_idx, pretrigger=PCI2ReadHeader(filename)
    nch=Number_of_channels
    voltage_scale=Max_voltage/32767
    fs = Sample_rate*1e3
    pretrigger_time=pretrigger/fs
    packet_size=8220
    packet_size_m=packet_size+2
    packet_size_block_str='4096*short'
    packet_size_block=4096
    signals=None

    if start_chnum is None:
        start_chnum=1
        end_chnum=Number_of_channels
    ext_chnum=end_chnum-start_chnum+1
    if start_time is not None:
        signals=np.zeros([round(fs*(end_time-start_time)),ext_chnum])


    for ii in range(start_chnum,end_chnum+1):
        f = open(filename,'rb')
        f.seek(Header_length+(packet_size*(ii-1))+28+(2*ii))
        if start_time is not None:
            if fs*start_time*2+math.floor(fs*start_time*2/(packet_size_block*2))*((packet_size_m*(Number_of_channels-1))+30) != 0:
                try :
                    f.seek(int(fs*start_time*2+math.floor(fs*start_time*2/(packet_size_block*2))*((packet_size_m*(Number_of_channels-1))+30)),1)
                except:
                    signals=None
                    t=None
                    fs=None
                    f.close()
                    return signals, t, fs, nch

            if(round((fs*start_time) % (packet_size_block))>0):
                # [channel, read_size] =f.read(round(packet_size_block-(fs*start_time)%packet_size_block)),packet_size_block_str,(packet_size_m*(Number_of_channels-1))+30)
                channel, read_size = read_packets(f, int(round(packet_size_block-((fs*start_time)%packet_size_block))), (packet_size_m * (Number_of_channels - 1)) + 30)

                if read_size<round(packet_size_block-((fs*start_time)%packet_size_block)):
                    signals = None
                    t = None
                    fs = None
                    f.close()
                    return signals, t, fs, nch
                f.seek((packet_size_m*(Number_of_channels-1))+30,1)
#                 [tmp_channel,read_size]=fread(fid,round(fs*(end_time-start_time)-(packet_size_block-mod(fs*start_time,packet_size_block))),packet_size_block_str,(packet_size_m*(Number_of_channels-1))+30);
                tmp_channel, read_size = read_packets(f, int(round(fs*(end_time-start_time)-(packet_size_block-((fs*start_time)%packet_size_block)))), (packet_size_m * (Number_of_channels - 1)) + 30)
                if read_size<round(fs*(end_time-start_time)-(packet_size_block-((fs*start_time)%packet_size_block))):
                    signals = None
                    t = None
                    fs = None
                    f.close()
                    return signals, t, fs, nch
                channel = np.hstack((channel, tmp_channel))
            else:
                # [channel,read_size]=fread(fid,round(fs*(end_time-start_time)),packet_size_block_str,(packet_size_m*(Number_of_channels-1))+30);
                channel, read_size=read_packets(f,int(round(fs*(end_time-start_time))), (packet_size_m*(Number_of_channels-1))+30)

                if read_size<round(fs*(end_time-start_time)):
                    signals = None
                    t = None
                    fs = None
                    f.close()
                    return signals, t, fs, nch
        else :
            # channel = fread(fid, packet_size_block_str, (packet_size_m * (Number_of_channels - 1)) + 30);
            channel, read_size = read_packets(f, -1, (packet_size_m * (Number_of_channels - 1)) + 30)

        f.close()

        if delay_idx<0:
            if ii==3 or ii==4:
                if delay_idx==0:
                    Ch = voltage_scale * channel.astype(np.float64)
                else:
                    Ch = voltage_scale*channel[:delay_idx].astype(np.float64)
            else:
                if delay_idx==0:
                    Ch = voltage_scale * channel.astype(np.float64)
                else:
                    Ch = voltage_scale*channel[-delay_idx:].astype(np.float64)
        else:
            if ii==1 or ii==2:
                if delay_idx==0:
                    Ch = voltage_scale * channel.astype(np.float64)
                else:
                    Ch = voltage_scale*channel[:-delay_idx].astype(np.float64)
            else:
                if delay_idx==0:
                    Ch = voltage_scale * channel.astype(np.float64)
                else:
                    Ch = voltage_scale*channel[delay_idx:].astype(np.float64)
        if signals is None:
            signals=np.zeros([len(Ch),nch])
        signals[:len(Ch),ii-start_chnum] = Ch

    if start_time is not None:
        t=np.arange(start_time,end_time,1/fs)
    else:
        t=np.arange(pretrigger/fs,(len(signals)+pretrigger)/fs,1/fs)

    return signals,t,fs, nch

def PCI2ReadHeader(filename):

    delay_idx_diff =0

    f = open(filename,'rb')
    Header = {'size_table':0, 'nch':0, 'sample_rate':0, 'trigger_mode':0, 'trigger_source':0, 'pretrigger':0,
              'maxvoltage':0, 'length_of_header':0}
    Header['size_table'] = struct.unpack('h',f.read(2))[0]
    f.seek(Header['size_table'], 1)

    Header['size_table']=struct.unpack('h',f.read(2))[0]
    f.seek(3, 1)
    Header['nch'] = struct.unpack('b',f.read(1))[0]
    f.seek(-4, 1)
    f.seek(Header['size_table'], 1)

    for i in range(Header['nch']):
        Header['size_table'] = struct.unpack('h',f.read(2))[0]
        f.seek(12, 1)
        Header['sample_rate'] = struct.unpack('h',f.read(2))[0]
        Header['trigger_mode'] = struct.unpack('h',f.read(2))[0]
        Header['trigger_source'] = struct.unpack('h',f.read(2))[0]
        Header['pretrigger'] = struct.unpack('h',f.read(2))[0]
        f.seek(2, 1)
        Header['maxvoltage'] = struct.unpack('h',f.read(2))[0]
        f.seek(-24, 1)
        f.seek(Header['size_table'], 1)
    k=1
    while Header['size_table'] != 8220:
        Header['size_table'] = struct.unpack('h',f.read(2))[0]

        if k == 7+Header['nch']*2:
            f.read(2)
            f.read(2)
            f.read(2)
            for i in range(Header['nch']):
                f.read(1)
                f.read(2)
                f.read(2)
                f.read(2)
                if i==0:
                    delay_idx=struct.unpack('q',f.read(8))[0]
                elif i==8:
                    delay_idx_diff=delay_idx-struct.unpack('q',f.read(8))[0]
                else:
                    f.read(8)
        else:
            f.seek(Header['size_table'], 1)
        k=k+1

    f.seek(-8222, 1)
    Header['length_of_header'] = f.tell()

    f.close()

    Number_of_channels=Header['nch']
    Sample_rate=Header['sample_rate']
    Max_voltage=Header['maxvoltage']
    Header_length=Header['length_of_header']
    Pretrigger=Header['pretrigger']

    return Number_of_channels,Sample_rate,Max_voltage, Header_length, delay_idx_diff, Pretrigger
