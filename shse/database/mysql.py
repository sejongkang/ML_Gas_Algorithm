"""
Author : Mingi Choi \n
Modified Date : 2018-05-30 \n
Description : Function for Database (MySQL)
"""
import logging
import MySQLdb
from pandas import DataFrame

# 데이터베이스(mysql) 클레스
class Mysql():
    """MySQL().__init__()

    Args:
        ip (str): 서버 IP
        port (int): 서버 PORT
        id (str): MySQL 접속 ID
        pw (str): MySQL 접속 PW
        db_name (str): 데이터베이스 이름
        auto_commit (bool): 자동 커밋

    Examples:
        >>> test = Mysql('localhost', 3306, 'root', 'offset01', 'mydb')
    """

    def __init__(self, ip, port, id, pw, db_name, auto_commit=True):
        self.__connect(ip, port, id, pw, db_name, auto_commit)

    def __del__(self):
        self.__disconnect()

    def __connect(self, ip, port, id, pw, db_name, auto_commit=True):
        try:
            self.conn = MySQLdb.connect(ip, id, pw, db_name, port, autocommit=auto_commit, charset='utf8')
        except Exception as e:
            msg = " \"" + self.connect.__name__ + "\" " + str(e)
            logging.error(msg)

    def __disconnect(self):
        if self.conn.open:
            self.conn.close()

    def is_connect(self) -> bool:
        """MySQL connection

        Returns:
            bool: 참, 거짓

        Examples:
            >>> test = Mysql('localhost', 3306, 'root', 'offset01', 'mydb')
            >>> test.is_connect()
            True
        """

        if self.conn.open:
            return True
        else:
            return False

    def commit(self):
        """Database coomit

        Examples:
            >>> test = Mysql('localhost', 3306, 'root', 'offset01', 'mydb')
            >>> test.commit()
        """

        try:
            self.conn.commit()
        except Exception as e:
            msg = " \"" + self.commit.__name__ + "\" " + str(e)
            logging.error(msg)

    def __create_cursor(self):
        try:
            self.cursor = self.conn.cursor(MySQLdb.cursors.DictCursor)
        except Exception as e:
            msg = " \"" + self.cursor.__name__ + "\" " + str(e)
            logging.error(msg)

    def drop_cursor(self):
        pass

    def call_procedure(self, proc_name, params=()) -> (str, list):
        """call mysql procedure

        Args:
            proc_name (str): 프로시저 이름
            params (list): 프로시저 파라미터

        Returns:
            DataFrame/False: 프로시저 수행 성공/실패

        Example:
            >>> test = Mysql('localhost', 3306, 'root', 'offset01', 'mydb')
            >>> test.call_procedure('test_procedure', ['1', '2', '3', '4', 5, '6'])
        """

        self.__create_cursor()
        try:
            self.cursor.callproc(procname=proc_name, args=params)
            data = self.__fetch_all()
        except Exception as e:
            msg = " \"" + self.call_procedure.__name__ + "\" " + str(e)
            logging.error(msg)
            self.cursor.close()
            return False
        else:
            self.cursor.close()
            return data

    def __fetch_all(self):
        results = self.cursor.fetchall()

        data_set = None
        for index, row in enumerate(results):
            if data_set is None:
                data_set = DataFrame(data=[row])
            else:
                data_set = data_set.append(DataFrame(data=[row]), ignore_index=True)

        return data_set

    def __fetch_one(self):
        """구현 중

        Returns:
        """

        pass

    def execute_query(self, query, params=None) -> (str, list):
        """execute query

        Args:
            query(str): SQL
            params(list): Parameter

        Returns:
            (int, DataFrame)/bool: (갯수, 결과)/실패

        Examples:
            >>> test = Mysql('localhost', 3306, 'root', 'offset01', 'mydb')
            >>> sql = 'select %s %s from user'
            >>> test.execute_query(sql, ['user_ui', 'user_passwd'])
            (1,               user_ui
            0  user_uiuser_passwd)
        """

        try:
            self.__create_cursor()
            return self.cursor.execute(query, params), self.__fetch_all()
        except:
            return False

    def insert(self, tbl_name, dataframe):
        """구현 중

        Returns:
        """

        pass

    def delete(self):
        """구현 중

        Returns:
        """

        pass

    def select(self):
        """구현 중

        Returns:
        """

        pass

    def update(self):
        """구현 중

        Returns:
        """

        pass


if __name__ == "__main__":
    a = Mysql('localhost', 3306, 'root', 'offset01', 'mydb', auto_commit=True)
    a.disconnect()
    print(a)