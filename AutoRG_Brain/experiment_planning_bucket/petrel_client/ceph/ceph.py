# -*- coding: utf-8 -*-


import re
from petrel_client.common.exception import InvalidClusterNameError, InvalidS3UriError, NoDefaultClusterNameError
from petrel_client.client_base import ClientBase


# (?:...)
# A non-capturing version of regular parentheses. Matches whatever regular expression is inside the parentheses, but the substring matched by the group cannot be retrieved after performing a match or referenced later in the pattern.

# *?, +?, ??
# The '*', '+', and '?' qualifiers are all greedy; they match as much text as possible. Sometimes this behaviour isn’t desired; if the RE <.*> is matched against <a> b <c>, it will match the entire string, and not just <a>. Adding ? after the qualifier makes it perform the match in non-greedy or minimal fashion; as few characters as possible will be matched. Using the RE <.*?> will match only <a>.

# re.I
# re.IGNORECASE
# Perform case-insensitive matching; expressions like [A-Z] will match lowercase letters, too. This is not affected by the current locale. To get this effect on non-ASCII Unicode characters such as ü and Ü, add the UNICODE flag.
# _S3_URI_PATTERN = re.compile(r'^(?:([^:]+):)?s3://([^/]+)/(.+?)/?$', re.I)

_S3_URI_PATTERN = re.compile(
    r'^(?:(?P<cluster>[^:]+):)?s3://(?P<bucket>[^/]+)/?(?P<key>(?:.+?)/?$)?', re.I)


class Ceph(ClientBase):

    @staticmethod
    def parse_uri(uri, ceph_dict, default_cluster=None):
        m = _S3_URI_PATTERN.match(uri)
        if m:
            cluster, bucket, key = m.group(
                'cluster'), m.group('bucket'), m.group('key')
            cluster = cluster or default_cluster
            if not cluster:
                raise NoDefaultClusterNameError(uri)

            try:
                client = ceph_dict[cluster]
                enable_cache = client.enable_cache()
                return cluster, bucket, key, enable_cache
            except KeyError:
                raise InvalidClusterNameError(cluster)
        else:
            raise InvalidS3UriError(uri)

    @staticmethod
    def create(cluster, conf, *args, **kwargs):
        fake = conf.get_boolean('fake')
        enable_s3_cpp = conf.get('boto').lower() in ('cpp', 'c++')
        enable_boto = (not enable_s3_cpp) and conf.get_boolean('boto')
        anonymous_access = (conf.get('access_key', None) is None) and (
            conf.get('secret_key', None) is None)

        if fake:
            from petrel_client.fake_client import FakeClient
            name = f'S3: {cluster}'
            return FakeClient(client_type='s3', conf=conf, name=name, **kwargs)
        elif enable_s3_cpp:
            from petrel_client.ceph.s3cpp.s3_cpp_client import S3CppClient
            return S3CppClient(cluster, conf, anonymous_access, *args, **kwargs)
        elif enable_boto:
            from petrel_client.ceph.s3.s3_client import S3Client
            return S3Client(cluster, conf, anonymous_access, *args, **kwargs)
        else:
            from petrel_client.ceph.librgw.rgw_client import RGWClient
            return RGWClient(cluster, conf, *args, **kwargs)

    def __init__(self, cluster, conf, *args, **kwargs):
        super(Ceph, self).__init__(*args, name=cluster, conf=conf, **kwargs)
        self.__enable_cache = conf.get_boolean(
            'enable_mc', False) or conf.get_boolean('enable_cache', False)

    def enable_cache(self):
        # 使用 __ 前缀使得 __enable_cache 变量成为私有变量
        # https://docs.python.org/3/tutorial/classes.html#private-variables
        # 将 enable_cache 类型定义为方法是为了以后可以动态计算是否需要做 cache
        return self.__enable_cache
