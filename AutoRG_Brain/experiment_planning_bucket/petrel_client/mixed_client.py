# -*- coding: utf-8 -*-

import logging
from os.path import expanduser, abspath

from petrel_client.ceph.ceph import Ceph
from petrel_client.cache.cache import Cache
from petrel_client.dfs.dfs import DFS
from petrel_client.common.config import Config
from petrel_client.common.log import init_log
from petrel_client.common import exception
from petrel_client.common.io_profile import Profiler
from petrel_client.common.io_retry import retry

if issubclass(str, bytes):
    def str_to_bytes(s):
        return s
else:
    import builtins

    def str_to_bytes(s):
        return builtins.bytes(s, 'utf-8')

LOG = logging.getLogger(__name__)


class MixedClient(object):
    def __init__(self, conf_path, **kwargs):
        conf_path = abspath(expanduser(conf_path))
        config = Config(conf_path)
        self._default_config = config.default()

        init_log(self._default_config)

        LOG.debug('init MixedClient, conf_path %s', conf_path)
        Profiler.set_default_conf(self._default_config)

        if any(conf.get_boolean('enable_mc') for _, conf in config.items()):
            cache_conf = config.get('cache', None) or config.get(
                'mc', None) or self._default_config
            self._cache = Cache.create(cache_conf, **kwargs)
        else:
            self._cache = None

        self._ceph_dict = {
            cluster: Ceph.create(cluster, conf)
            for cluster, conf in config.items() if cluster.lower() not in ('dfs', 'cache', 'mc')
        }

        dfs_conf = config.get('dfs', self._default_config)
        self._dfs = DFS.create(dfs_conf)

        self._default_cluster = self._default_config.get(
            'default_cluster', None)
        self._count_disp = self._default_config.get_int('count_disp')
        self._get_retry_max = self._default_config.get_int('get_retry_max')

    def ceph_parse_uri(self, uri, content):
        cluster, bucket, key, enable_cache = Ceph.parse_uri(
            uri, self._ceph_dict, self._default_cluster)

        def io_fn(**kwargs):
            client = self._ceph_dict[cluster]
            if content is not None:
                return client.put_with_info(cluster, bucket, key, content, **kwargs)
            else:
                return client.get_with_info(cluster, bucket, key, **kwargs)

        return enable_cache, io_fn

    def cache_parse_uri(self, uri, content):
        key = Cache.parse_uri(uri)

        def io_fn(**kwargs):
            if content is not None:  # todo add info
                return self._cache.put(key, content), None
            else:
                return self._cache.get(key), None
        return False, io_fn

    def dfs_parse_uri(self, uri, content):
        file_path = DFS.parse_uri(uri)

        def io_fn(**kwargs):  # todo add info
            if content is not None:
                return self._dfs.put(file_path, content, **kwargs), None
            else:
                return self._dfs.get(file_path, **kwargs), None

        return self._dfs.enable_cache(), io_fn

    def prepare_io_fn(self, uri, content=None):
        try:
            return self.ceph_parse_uri(uri, content)
        except exception.InvalidS3UriError:
            pass

        try:
            return self.cache_parse_uri(uri, content)
        except exception.InvalidMcUriError:
            pass

        try:
            return self.dfs_parse_uri(uri, content)
        except exception.InvalidDfsUriError:
            pass

        raise exception.InvalidUriError(uri)

    def _get_with_info(self, uri, **kwargs):  # returns (data, info)
        no_cache = kwargs.get('no_cache', False)
        update_cache = kwargs.get('update_cache', False)

        if no_cache and update_cache:
            raise ValueError(
                'arguments "update_cache" and "no_cache" conflict with each other')

        enable_cache, get_fn = self.prepare_io_fn(uri)
        enable_cache = self._cache and enable_cache and (not no_cache)
        cache_retry_times = 3
        cache_value = None

        if enable_cache and (not update_cache):
            for _ in range(cache_retry_times):
                cache_should_retry = False
                try:
                    cache_value = self._cache.get(uri, **kwargs)
                except exception.ObjectNotFoundError:
                    pass
                except exception.CacheError as err:
                    self._cache.log.debug(err)
                    if isinstance(err, exception.RetriableError):
                        cache_should_retry = True
                except Exception as err:
                    LOG.error(err)
                finally:
                    if not cache_should_retry:
                        break

        if cache_value is not None:
            return cache_value, None

        content, info = get_fn(**kwargs)

        if enable_cache:
            for _ in range(cache_retry_times):
                cache_should_retry = False
                try:
                    self._cache.put(uri, content)  # todo 如果有异常，是否该忽略？
                except exception.CacheError as err:
                    self._cache.log.debug(err)
                    if isinstance(err, exception.RetriableError):
                        cache_should_retry = True
                except Exception as err:
                    LOG.error(err)
                finally:
                    if not cache_should_retry:
                        break

        return content, info

    # 所有的异常在此处处理
    def get_with_info(self, uri, **kwargs):
        @retry('get', exceptions=(Exception,), raises=(exception.ResourceNotFoundError, NotImplementedError), tries=self._get_retry_max)
        def do_get_with_info(self, uri, **kwargs):
            try:
                return self._get_with_info(uri, **kwargs)
            except exception.NoSuchBucketError as err:
                LOG.warning(err)
            except exception.ObjectNotFoundError as err:
                LOG.debug(err)
            except exception.AccessDeniedError as err:
                LOG.warning((err, uri))

            return None, None

        return do_get_with_info(self, uri, **kwargs)

    def create_bucket(self, uri, **kwargs):
        cluster, bucket, key, _ = Ceph.parse_uri(
            uri, self._ceph_dict, self._default_cluster)
        if key is not None:
            raise exception.InvalidBucketUriError(uri)
        return self._ceph_dict[cluster].create_bucket(bucket)

    def isdir(self, uri, **kwarg):
        try:
            cluster, bucket, key, enable_cache = Ceph.parse_uri(
                uri, self._ceph_dict, self._default_cluster)
            client = self._ceph_dict[cluster]
            isdir_fn = getattr(client, 'isdir')
            return isdir_fn(bucket, key)
        except exception.InvalidS3UriError:
            LOG.error(f'Invalid S3 URI: ${uri}')
            raise
        except AttributeError:
            LOG.warning('please set boto = True to use this feature')
            raise

    def list(self, uri, **kwarg):
        try:
            cluster, bucket, key, enable_cache = Ceph.parse_uri(
                uri, self._ceph_dict, self._default_cluster)
            client = self._ceph_dict[cluster]
            list_fn = getattr(client, 'list')
            return list_fn(bucket, key, **kwarg)
        except exception.InvalidS3UriError:
            LOG.error(f'Invalid S3 URI: ${uri}')
            raise
        except AttributeError:
            LOG.warning('please set boto = True to use this feature')
            raise

    def get_file_iterator(self, uri):
        try:
            cluster, bucket, key, enable_cache = Ceph.parse_uri(
                uri, self._ceph_dict, self._default_cluster)
            client = self._ceph_dict[cluster]
            file_iterator = getattr(client, 'get_file_iterator')
            return file_iterator(bucket, key)
        except exception.InvalidS3UriError:
            LOG.error('only support ceph')
            raise
        except AttributeError:
            LOG.warning('please set boto = True to use this feature')
            raise

    def put_with_info(self, uri, content, **kwargs):
        if isinstance(content, str):
            content = str_to_bytes(content)

        _enable_cache, put_fn = self.prepare_io_fn(uri, content)

        update_cache = self._cache and kwargs.get('update_cache', False)

        result, info = put_fn(**kwargs)

        if update_cache:
            self._cache.put(uri, content)

        return result, info

    def contains(self, uri):
        cluster, bucket, key, _ = Ceph.parse_uri(
            uri, self._ceph_dict, self._default_cluster)
        client = self._ceph_dict[cluster]
        return client.contains(cluster, bucket, key)

    def delete(self, uri):
        cluster, bucket, key, _ = Ceph.parse_uri(
            uri, self._ceph_dict, self._default_cluster)
        client = self._ceph_dict[cluster]
        return client.delete(cluster, bucket, key)

    def generate_presigned_url(self, uri, client_method='get_object', expires_in=3600):
        cluster, bucket, key, _ = Ceph.parse_uri(
            uri, self._ceph_dict, self._default_cluster)
        client = self._ceph_dict[cluster]
        return client.generate_presigned_url(cluster, bucket, key, client_method, expires_in)

    def generate_presigned_post(self, uri, fields=None, conditions=None, expires_in=3600):
        cluster, bucket, key, _ = Ceph.parse_uri(
            uri, self._ceph_dict, self._default_cluster)
        client = self._ceph_dict[cluster]
        return client.generate_presigned_post(cluster, bucket, key, fields, conditions, expires_in)

    def set_count_disp(self, count_disp):
        Profiler.set_count_disp(count_disp)
