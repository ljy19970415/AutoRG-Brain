import functools
from urllib.parse import urlparse
import logging
import hashlib

from petrel_client.common.io_profile import profile
from petrel_client.ceph.s3cpp.pys3client import PyS3Client, S3Error, init_api, shutdown_api
from petrel_client.ceph.ceph import Ceph
from petrel_client.common import exception

LOG = logging.getLogger(__name__)

EXCEPTION_MAP = {
    'ACCESS_DENIED': exception.AccessDeniedError,
    'NO_SUCH_BUCKET': exception.NoSuchBucketError,
    'NO_SUCH_KEY': exception.NoSuchKeyError,
    'RESOURCE_NOT_FOUND': exception.ResourceNotFoundError,
    'SIGNATURE_DOES_NOT_MATCH': exception.SignatureNotMatchError,
    'INVALID_ACCESS_KEY_ID': exception.InvalidAccessKeyError,
    'NETWORK_CONNECTION': exception.NetworkConnectionError,
}

S3_CPP_ENV = None


class S3CppEnv(object):
    def __init__(self, log_level):
        LOG.debug('S3CppEnv init')
        init_api(log_level)

    def __del__(self):
        # LOG.debug('S3CppEnv del') del 阶段log抛异常
        shutdown_api()


def get_s3_cpp_env(log_level):
    global S3_CPP_ENV
    if S3_CPP_ENV is None:
        S3_CPP_ENV = S3CppEnv(log_level)
    return S3_CPP_ENV


def wrap_error(fn):
    @functools.wraps(fn)
    def new_fn(self, cluster, bucket, key, *args, **kwargs):
        try:
            return fn(self, cluster, bucket, key, *args, **kwargs)
        except S3Error as err:
            err_type = EXCEPTION_MAP.get(err.error_name, None)
            if err_type:
                new_err = err_type(cluster, bucket, key)
            elif err.error_message:
                new_err = exception.S3ClientError(
                    err.error_name, err.error_message)
            else:
                new_err = exception.S3ClientError(err.error_name)

            new_err.__traceback__ = err.__traceback__
            raise new_err from None

    return new_fn


class S3CppClient(Ceph):
    def __init__(self, cluster, conf, anonymous_access, *args, **kwargs):
        # 如果初始化出现异常，将会调用 __del__ ，这里先赋值避免 __del__ 出现逻辑错误
        self._client = None
        self._env = None

        endpoint_url = conf['endpoint_url']
        if '://' not in endpoint_url:
            endpoint_url = 'http://' + endpoint_url
        parse_result = urlparse(endpoint_url)
        s3_args = {
            # AWS CPP SDK 中 ak 和 sk 为空时表示匿名访问
            'ak': b'' if anonymous_access else conf['access_key'].encode('utf-8'),
            'sk': b'' if anonymous_access else conf['secret_key'].encode('utf-8'),

            'endpoint': parse_result.netloc.encode('utf-8'),
            'enable_https': parse_result.scheme == 'https',
            'verify_ssl': conf.get_boolean('verify_ssl', False),
            'use_dual_stack': False,
        }

        super(S3CppClient, self).__init__(cluster, conf, *args, **kwargs)
        self._cluster = cluster
        self._conf = conf

        s3_cpp_log_level = conf.get('s3_cpp_log_level')
        self._env = get_s3_cpp_env(s3_cpp_log_level)
        self._client = PyS3Client(**s3_args)

    def __del__(self):
        del self._client
        del self._env

    @profile('get')
    @wrap_error
    def get_with_info(self, cluster, bucket, key, **kwargs):
        info = {}

        unsupported_ops = [k for k, v in kwargs.items() if k in (
            'enable_stream', 'enable_etag') and v]
        if unsupported_ops:
            raise NotImplementedError(unsupported_ops)

        if isinstance(bucket, str):
            bucket = bucket.encode('utf-8')
        if isinstance(key, str):
            key = key.encode('utf-8')
        data = self._client.get_object(bucket, key)

        enable_md5 = kwargs.get('enable_md5', False)
        if enable_md5:
            info['md5'] = hashlib.md5(data).hexdigest()

        return data, info

    @profile('put')
    @wrap_error
    def put_with_info(self, cluster, bucket, key, body, **kwargs):
        info = {}  # todo
        if not isinstance(body, (bytes, bytearray)):
            raise NotImplementedError(f'unsupported type f{type(body)}')
        if isinstance(bucket, str):
            bucket = bucket.encode('utf-8')
        if isinstance(key, str):
            key = key.encode('utf-8')
        return self._client.put_object(bucket, key, body), info
