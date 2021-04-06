# -*- coding: utf-8 -*-
import os
import socket
import struct

import numpy as np
from framework.common.common_func import *
from framework.common.common_log import CommonLogger

LOG = CommonLogger.get_logger()


class Sender():
    def __init__(self, mem_pool_path, config_id, ip="128.0.0.0", use_zmq=False):
        self.m_mem_pool_path = mem_pool_path
        self.config_id = int(config_id)
        self.config_id += int(socket.ntohl(struct.unpack("I", socket.inet_aton(str(ip)))[0]))
        self.use_zmq = use_zmq
        if not self._init_mempool_send():
            LOG.error("init mempool send failed")

    @log_time("send_data")
    def send_data(self, send_sample):
        if self.m_client is None:
            LOG.error("create new client")
            self._init_mempool_send()
        while True:
            ret = self.m_client.send_data(send_sample)
            if not ret:
                del self.m_client
                self._create_client()
                LOG.error("send data failed, create new client")
            else:
                break
        ret = self.m_client.recv_data()
        if not ret:
            del self.m_client
            return True
        return True

    def _init_mempool_send(self):
        if os.path.exists(self.m_mem_pool_path):
            fin = open(self.m_mem_pool_path, "r")
            lines = fin.readlines()
            fin.close()
            if not lines:
                self.m_client = None
                return True
            rand_line = int(np.random.randint(len(lines)))
            self.m_mem_pool_ip, self.m_mem_pool_port = lines[rand_line].strip().split(":")
            LOG.info("m_mem_pool_ip port: %s %s " % (self.m_mem_pool_ip, self.m_mem_pool_port))
            self._create_client()
            return True
        else:
            LOG.error("hasn't mem_pool.host_list")
            self.m_client = None
            return False

    def _create_client(self):
        self.m_client = Client(self.m_mem_pool_ip, self.m_mem_pool_port, self.use_zmq)
