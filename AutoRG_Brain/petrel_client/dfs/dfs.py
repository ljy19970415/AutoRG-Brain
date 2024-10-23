import logging
import re
import socket

from petrel_client.client_base import ClientBase
from petrel_client.common.io_profile import profile
from petrel_client.common import exception

LOG = logging.getLogger(__name__)


class DFS(ClientBase):

    @staticmethod
    def parse_uri(uri):
        # todo check if it is a valid path
        return re.sub('^file://', '/', uri)

    @staticmethod
    def create(conf, *args, **kwargs):
        fake = conf.get_boolean('fake')
        if fake:
            from petrel_client.fake_client import FakeClient
            name = f'DFS: {socket.gethostname()}'
            return FakeClient(client_type='dfs', conf=conf, name=name, **kwargs)
        else:
            return DFS(conf, *args, **kwargs)

    def __init__(self, conf, *args, **kwargs):
        hostname = socket.gethostname()

        super(DFS, self).__init__(*args, name=hostname, conf=conf, **kwargs)
        self._enable_cache = conf.get_boolean(
            'enable_mc', False) or conf.get_boolean('enable_cache', False)

    @profile('get')
    def get(self, file_path, **kwargs):
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except FileNotFoundError as err:
            raise exception.ObjectNotFoundError(err)
        except Exception as err:
            raise exception.ClientError(err)

    def put(self, file_path, content, **kwargs):
        try:
            with open(file_path, 'wb') as f:
                return f.write(content)
        except Exception as err:
            raise exception.ClientError(err)

    def enable_cache(self):
        return self._enable_cache
