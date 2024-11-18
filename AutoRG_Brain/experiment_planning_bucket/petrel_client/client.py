# -*- coding: utf-8 -*-

import threading
import logging
import functools
import os

from petrel_client.mixed_client import MixedClient

LOG = logging.getLogger(__name__)
thread_local_client = threading.local()

DEFAULT_CONF_PATH = '~/petreloss.conf'


class Client(object):

    def __init__(self, conf_path=None, *args, **kwargs):
        self._conf_path = conf_path or DEFAULT_CONF_PATH
        self.kwargs = kwargs

        # 用户在调用 Client() 就实例化 GenericClient
        # 如果该 GenericClient 实例化失败就抛异常
        # 避免在 put/get 的时候才开始抛异常
        # 此外，如果用户使用log，multiprocessing-logging 需要在Client创建的进程中初始化
        self._get_local_client()

    def _get_local_client(self):
        current_pid = os.getpid()
        client, client_pid = getattr(
            thread_local_client,
            self._conf_path,
            (None, None)
        )
        if current_pid != client_pid:
            client = MixedClient(self._conf_path, **self.kwargs)
            setattr(
                thread_local_client,
                self._conf_path,
                (client, current_pid)
            )
        return client

    def get_with_info(self, uri, **kwargs):
        return self._get_local_client().get_with_info(uri, **kwargs)

    def get(self, *args, **kwargs):
        data, _ = self.get_with_info(*args, **kwargs)
        return data

    def list(self, *args, **kwargs):
        client = self._get_local_client()
        return client.list(*args, **kwargs)

    def isdir(self, uri):
        client = self._get_local_client()
        return client.isdir(uri)

    def get_file_iterator(self, uri):
        try:
            client = self._get_local_client()
            file_iterator = client.get_file_iterator(uri)
            return file_iterator
        except Exception as e:
            LOG.error('get file generator error {0}'.format(e))
            raise

    def put_with_info(self, uri, content, **kwargs):
        return self._get_local_client().put_with_info(uri, content, **kwargs)

    def put(self, *args, **kwargs):
        result, _ = self.put_with_info(*args, **kwargs)
        return result

    def contains(self, *args, **kwargs):
        return self._get_local_client().contains(*args, **kwargs)

    def delete(self, *args, **kwargs):
        self._get_local_client().delete(*args, **kwargs)

    def generate_presigned_url(self, *args, **kwargs):
        return self._get_local_client().generate_presigned_url(*args, **kwargs)

    def generate_presigned_post(self, *args, **kwargs):
        return self._get_local_client().generate_presigned_post(*args, **kwargs)

    def create_bucket(self, *args, **kwargs):
        return self._get_local_client().create_bucket(*args, **kwargs)

    Get = get

    GetAndUpdate = get_and_update = functools.partialmethod(
        get, update_cache=True)

    def set_count_disp(self, count_disp):
        self._get_local_client().set_count_disp(count_disp)
