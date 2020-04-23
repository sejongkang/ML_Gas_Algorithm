import pymysql
import threading
import time
import torch
import numpy as np
from shse.mlearn.utils import load_model
from sklearn.externals import joblib
import warnings

class Device:
    def __init__(self,dev_serial):
        self.dev_serial = dev_serial
        self.pre_type = 'Normal'
        self.column_name = ['ppm','ppm2','ppm3','ppm4','ppm6','ppm6']
        self.timer_flag = False
        timer_thread = threading.Thread(target=self.SetTimer, args=())
        timer_thread.daemon = True  # SubThread 선언
        timer_thread.start()

    def Do(self):
        try:
            db_data = self.GetData()
            if db_data!=():
                data = db_data[0]
                data_idx = int(data[0])
                x_data = list(data[1:7])
                is_enroll = data[7]
                if is_enroll == 0:
                    x_data_ratio = self.get_Ratio(x_data)
                    x_scaled = scaler_model.transform([x_data_ratio])
                    x = torch.Tensor(x_scaled).float()
                    x = x.unsqueeze(dim=0)
                    if gpu:
                        result = clf_model.forward(x.cuda())
                    else:
                        result = clf_model.forward(x)
                    self.classify = torch.argmax(result, dim=-1)
                    self.classify = self.classify.squeeze().tolist()
                    self.type = class_name[self.classify]
                    self.ppm = self.GetPPm(self.classify-1, x_data)
                    print(self.dev_serial, "type :", self.type)
                    if (self.pre_type != self.type):
                        self.timer_flag = True
                    if self.timer_flag == False:
                        if self.type == 'Normal':
                            self.PutData(data_idx)
                        else:
                            self.PutData(data_idx, type = self.column_name[self.classify-1], value = self.ppm)
                        # print(self.dev_serial, "DB Insert result")
                    else:
                        self.PutData(data_idx, loading=True)
                        # print(self.dev_serial, "DB Insert loading")
                    self.pre_type = self.type
                else:
                    print(self.dev_serial+" pass")
        except Exception as e:
            print("Error : ", e)

    def GetData(self):
        conn = pymysql.connect(host=Host, port=Port, database=Database, user=User, password=Password)
        with conn.cursor() as cursor:
            sql = "SELECT  `idx`, `H2`,  `VOC`,  `Methyl`,  `LP`,  `Solvent`,  `NH3`, `isEnroll` FROM `gas`.`gas_log` WHERE gas_module_idx = '"+self.dev_serial+"' ORDER BY `idx` DESC LIMIT 1;"
            cursor.execute(sql)
            data = cursor.fetchall()
        conn.close()
        return data

    def PutData(self,log_idx, type = 0, type2 = 0, value=0, value2=0, loading=False):
        conn = pymysql.connect(host=Host, port=Port, database=Database, user=User, password=Password)
        with conn.cursor() as cursor:
            if loading:
                sql = 'insert into gas.result_info (log_idx,loading) values (%s,%s);'
                cursor.execute(sql, (log_idx, loading));
            else:
                if type == 0:
                    sql = 'insert into gas.result_info (log_idx) values (%s);'
                    cursor.execute(sql, (log_idx));
                else :
                    if type2 == 0:
                        sql = 'insert into gas.result_info (log_idx,' + type + ') values (%s,%s);'
                        cursor.execute(sql, (log_idx, value));
                    else :
                        sql = 'insert into gas.result_info (log_idx,' + type + ',' + type2 + ') values (%s,%s,%s);'
                        cursor.execute(sql, (log_idx, value, value2));
            sql = 'UPDATE gas_log SET isEnroll = %s WHERE idx = %s'
            cursor.execute(sql, (True, log_idx))
        conn.commit()
        conn.close()

    def GetPPm(self, type, x_data):
        x = torch.Tensor(x_data).float()
        x = np.reshape(x, (-1, 1, 6))
        if gpu:
            result = round(reg_model[type].forward(x.cuda()).tolist()[0][0], 1)
        else:
            result = round(reg_model[type].forward(x).tolist()[0][0], 1)
        return result

    def SetTimer(self):
        while (1):
            if self.timer_flag:
                # print(self.dev_serial,"Delay Start")
                for k in range(60):
                    # print(60 - k, "sec")
                    time.sleep(1)
                self.timer_flag = False
                # print(self.dev_serial, "Delay End")

    def get_Ratio(self,sensorValue):
        sensorValue = np.array(sensorValue)
        for i in range(6):
            if sensorValue[i] <= 0:
                sensorValue[i] = 1
        featlist = []
        for i in range(6):
            for j in range(6):
                if i != j :
                    featlist.append(sensorValue[j] / sensorValue[i])
        return np.transpose(featlist)

def DevRun(device):
    while(1):
        # try:
        device.Do()
        # except Exception as ex:
        #     print(ex)
        time.sleep(30)

if __name__ == '__main__':
    warnings.filterwarnings(action='ignore')

    gpu = torch.cuda.is_available()  # Check Support CUDA

    # Host = '192.168.2.2'
    # Port = 3306
    # Database = 'gas'
    # User = 'root'
    # Password = 'ubimicro'

    # Host = '203.250.78.169'
    # Port = 3307
    # Database = 'gas'
    # User = 'root'
    # Password = 'offset01'

    Host = '106.252.240.216'
    Port = 23306
    Database = 'gas'
    User = 'root'
    Password = 'ubimicro'

    print("- Host :",Host)
    print("- Port :", Port)
    print("- DB :", Database)
    print("- GPU Use :", gpu)
    print("")
    class_name = ['Normal','H2S','NH3','CH3SH','CO','CH4']
    clf_name = 'model/Classify/Clf_single.pkl'
    reg_name = ['model/Regression/H2S_Reg.pkl', 'model/Regression/NH3_Reg.pkl', 'model/Regression/CH3SH_Reg.pkl',
                'model/Regression/CO_Reg.pkl', 'model/Regression/CH4_Reg.pkl']
    scaler_model = joblib.load('model/Classify/single_scaler.pkl')
    print("- Model Loaded")

    reg_model = []

    if gpu:
        clf_model = load_model(clf_name)
        clf_model.train(False)
        print("- DNN CLF Model Loaded")
        for i in range(len(reg_name)):
            tmp_reg = load_model(reg_name[i])
            tmp_reg.train(False)
            reg_model.append(tmp_reg)
        print("- DNN REG Model Loaded")
    else:
        clf_model = load_model(clf_name)
        print("- DNN CLF Model Loaded")
        for i in range(len(reg_name)):
            tmp_reg = load_model(reg_name[i], map_location='cpu')
            tmp_reg.train(False)
            reg_model.append(tmp_reg)
        print("- DNN REG Model Loaded")

    dev = [Device('010000FFFF000030'),Device('010000FFFF00004A'),
           Device('010000FFFF000032'),Device('010000FFFF000033'),
           Device('010000FFFF000034')]
    print("")

    dev_thread = threading.Thread(target=DevRun, args=(dev[0],))
    dev_thread.daemon = True  # SubThread 선언
    dev_thread.start()
    print("- Device1 Thread Run")
    dev_thread2 = threading.Thread(target=DevRun, args=(dev[1],))
    dev_thread2.daemon = True  # SubThread 선언
    dev_thread2.start()
    # print("- Device2 Thread Run")
    # dev_thread3 = threading.Thread(target=DevRun, args=(dev[2],))
    # dev_thread3.daemon = True  # SubThread 선언
    # dev_thread3.start()
    # print("- Device3 Thread Run")
    # dev_thread4 = threading.Thread(target=DevRun, args=(dev[3],))
    # dev_thread4.daemon = True  # SubThread 선언
    # dev_thread4.start()
    # print("- Device4 Thread Run")
    # dev_thread5 = threading.Thread(target=DevRun, args=(dev[4],))
    # dev_thread5.daemon = True  # SubThread 선언
    # dev_thread5.start()
    # print("- Device5 Thread Run")
    # print("")
    dev_thread.join()

