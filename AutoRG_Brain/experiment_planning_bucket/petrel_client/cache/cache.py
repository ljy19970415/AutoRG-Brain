import logging
import socket
import re
import sys
from petrel_client.client_base import ClientBase
from petrel_client.common.exception import InvalidMcUriError

LOG = logging.getLogger(__name__)

_MC_URI_PATTERN = re.compile(r'^mc://(.+)')

import_map = {'memcached': 'petrel_client.cache.mc.mc.MC'}


class Cache(ClientBase):
    @staticmethod
    def parse_uri(uri):
        m = _MC_URI_PATTERN.match(uri)
        if m:
            return m.group(1)
        else:
            raise InvalidMcUriError(uri)

    @staticmethod
    def get_engine_cls(engine_type):
        import_name = import_map.get(engine_type)
        module_name, callable_name = import_name.rsplit('.', 1)
        __import__(module_name)
        module = sys.modules[module_name]
        return getattr(module, callable_name)

    @staticmethod
    def create(conf, *args, **kwargs):
        fake = conf.get_boolean('fake')
        if fake:
            from petrel_client.fake_client import FakeClient
            name = f'MC: {socket.gethostname()}'
            return FakeClient(client_type='mc', conf=conf, name=name, **kwargs)

        engine_type = conf.get('cache_engine', 'memcached')
        try:
            engine_cls = Cache.get_engine_cls(engine_type)
            instance = engine_cls(conf, *args, **kwargs)
            if not hasattr(instance, 'log'):
                setattr(instance, 'log', LOG)
            return instance
        except Exception as err:
            LOG.warn('can not init cache client')
            LOG.exception(err)
        return None

    def __init__(self, *args, **kwargs):
        super(Cache, self).__init__(*args, **kwargs)
