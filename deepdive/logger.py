#logger.py

import logging.handlers

# Logger 인스턴스 생성 및 로그 레벨 설정
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

# formatter 생성
formatter = logging.Formatter('[%(filename)s:%(lineno)s]%(asctime)s>%(message)s')

# FileHandler와 StreamHandler 생성
fileMaxByte = 1024 * 1024 * 100    #100MB
fileHandler = logging.handlers.RotatingFileHandler('./log/my.log', maxBytes=fileMaxByte,
                                                   backupCount=10, encoding='utf-8')
streamHandler = logging.StreamHandler()

# handler에 formatter 사용
fileHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)

# handler를 logging에 추가
logger.addHandler(fileHandler)
logger.addHandler(streamHandler)