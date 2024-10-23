# -*- coding: utf-8 -*-
import hashlib

from petrel_client.common.exception import ConfigKeyValueError

_SUPPORTED_TYPES = ('blake2b', 'blake2s', 'md5', 'pbkdf2_hmac', 'sha1', 'sha224', 'sha256',
                    'sha384', 'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512', 'sha512', 'shake_128', 'shake_256')


def get_hash_fn(hash_type):
    if hash_type in _SUPPORTED_TYPES:
        return getattr(hashlib, hash_type)
    else:
        raise ConfigKeyValueError(f"'{hash_type}' is not a valid hash type.")


def to_bytes(key):
    if isinstance(key, str):
        key = key.encode('utf-8')
    else:
        assert isinstance(key, bytes)
    return key


def hexdigest(key, hash_fn):
    key = to_bytes(key)
    m = hash_fn()
    m.update(key)
    return m.hexdigest()
