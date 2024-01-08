import os
import sys


class Logger(object):
    """日志输出"""
    def __init__(self, filename='logs.log', stream=sys.stdout, path='./outpus/'):
        """初始化日志"""
        log_file = os.path.join(path, filename)
        self.terminal = stream
        self.log = open(log_file, 'a+')

    def write(self, message):
        """写入日志"""
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """刷新日志"""
        self.log.flush()
        self.terminal.flush()
