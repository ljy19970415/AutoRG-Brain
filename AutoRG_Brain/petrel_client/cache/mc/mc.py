import functools
import logging
from functools import partial
from collections import defaultdict

from petrel_client.cache.cache import Cache
from petrel_client.common.io_profile import profile
from petrel_client.common import exception
from petrel_client.cache.mc.petrel_pymc import McClient
from petrel_client.common import hash

LOG = logging.getLogger(__name__)

_STATUS_SUCCESS = 'SUCCESS'
_STATUS_NOT_FOUND = 'NOT FOUND'

_MAX_KEY_SIZE = 250
_ITEM_SIZE_RESERVED = 128

_EXCEPTION_MAP = defaultdict(lambda: exception.McClientError, {
    'A TIMEOUT OCCURRED': exception.McTimeoutOccur,
    'CONNECTION FAILURE': exception.McConnFailed,
    'FAILURE': exception.McServerDisable,
    'CLIENT ERROR': exception.McServerDisable,
    'SERVER ERROR': exception.McServerDisable,
    'ERROR was returned by server': exception.McServerDisable,
    'SYSTEM ERROR': exception.McServerFailed,
    'A KEY LENGTH OF ZERO WAS PROVIDED': exception.McBadKeyProvided,
    'A BAD KEY WAS PROVIDED/CHARACTERS OUT OF RANGE': exception.McBadKeyProvided,
    'SERVER IS MARKED DEAD': exception.McServerDead,
    'ITEM TOO BIG': exception.McObjectSizeExceed,
    'SERVER HAS FAILED AND IS DISABLED UNTIL TIMED RETRY': exception.McServerFailed,
})


def wrap_io(fn):
    @functools.wraps(fn)
    def new_fn(self, key, *args, **kwargs):
        if self.mc_key_cb:
            key = self.mc_key_cb(key)
        self.check_key_size(key)

        value, status = fn(self, key, *args, **kwargs)

        if status == _STATUS_SUCCESS:
            return value
        elif status == _STATUS_NOT_FOUND:
            raise exception.McObjectNotFoundError(key)
        else:
            server, _ = self._mc.get_server(key)
            raise _EXCEPTION_MAP[status](key, status, server)

    return new_fn


class MC(Cache):
    def __init__(self, conf, *args, **kwargs):
        mc_server_list_path = conf['mc_server_list_path']
        mc_client_config_path = conf['mc_client_config_path']
        debug_mc = conf.get_boolean('debug_mc')
        if debug_mc:
            LOG.setLevel(logging.DEBUG)
        else:
            LOG.setLevel(logging.WARNING)

        self.log = LOG
        LOG.debug('init MC, server list path: %s, client config path: %s',
                  mc_server_list_path, mc_client_config_path)
        super(MC, self).__init__(*args, conf=conf, **kwargs)

        self._mc = McClient.GetInstance(
            mc_server_list_path, mc_client_config_path)
        self._max_item_size = self._mc.max_item_size() - _MAX_KEY_SIZE - \
            _ITEM_SIZE_RESERVED
        self._max_key_size = _MAX_KEY_SIZE

        mc_key_cb = kwargs.get('mc_key_cb', None) or conf.get('mc_key_cb')
        if mc_key_cb == 'identity':
            self.mc_key_cb = None
        elif isinstance(mc_key_cb, str):
            hash_fn = hash.get_hash_fn(mc_key_cb)
            self.mc_key_cb = partial(hash.hexdigest, hash_fn=hash_fn)
            LOG.debug('mc: using mc_key_cb %s', mc_key_cb)
        elif not callable(mc_key_cb):
            raise Exception("argument 'mc_key_cb' should be callable.")
        else:
            self.mc_key_cb = mc_key_cb
            LOG.debug('mc: using user defined mc_key_cb')

    def check_key_size(self, key):
        if isinstance(key, str):
            key_len = len(key.encode('utf-8'))
        elif isinstance(key, bytes):
            key_len = len(key)
        else:
            raise Exception(
                'mc key type is not supported: {}, value: {}'.format(type(key), key))

        if key_len > self._max_key_size:
            raise exception.McKeySizeExceed(
                'size of key must <= {}'.format(self._max_key_size), key)

    @profile('get')
    @wrap_io
    def get(self, key, **kwargs):
        return self._mc.get(key)

    @profile('put')
    @wrap_io
    def put(self, key, content, **kwargs):
        size = len(content)
        if size > self._max_item_size:
            raise exception.McObjectSizeExceed(
                key, 'size of object must <= {}, actual size: {}'.format(self._max_item_size, size))
        status = self._mc.set(key, content)
        return size, status
