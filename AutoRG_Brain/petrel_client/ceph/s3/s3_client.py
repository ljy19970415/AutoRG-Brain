# -*- coding: utf-8 -*-

import logging
import hashlib

import boto3
from botocore.exceptions import ClientError as BotoClientError
from botocore.client import Config
from botocore import UNSIGNED

from petrel_client.ceph.ceph import Ceph
from petrel_client.common.io_profile import profile
from petrel_client.common.exception import NoSuchBucketError, NoSuchKeyError, S3ClientError, AccessDeniedError
from .generator import FileIterator

LOG = logging.getLogger(__name__)


class S3Client(Ceph):

    def __init__(self, cluster, conf, anonymous_access, *args, **kwargs):
        if anonymous_access:
            s3_args = {
                'config': Config(signature_version=UNSIGNED)
            }
        else:
            s3_args = {
                'aws_access_key_id': conf['access_key'],
                'aws_secret_access_key': conf['secret_key']
            }

        s3_args['endpoint_url'] = conf['endpoint_url']
        s3_args['verify'] = conf.get_boolean('verify_ssl', False)

        super(S3Client, self).__init__(cluster, conf, *args, **kwargs)
        self._cluster = cluster
        self._conf = conf
        self._session = boto3.session.Session()
        self._s3_resource = self._session.resource(
            's3',
            **s3_args
        )

    @profile('get')
    def get_with_info(self, cluster, bucket, key, **kwargs):
        enable_etag = kwargs.get('enable_etag', False)
        enable_stream = kwargs.get('enable_stream', False)
        info = {}
        assert self._cluster == cluster
        try:
            obj = self._s3_resource.Object(bucket, key).get()
            content = obj['Body']
            if not enable_stream:
                content = content.read()
            if enable_etag:
                info['etag'] = obj['ETag'].strip('"')
            return content, info
        except BotoClientError as err:
            if type(err).__name__ == 'NoSuchKey':
                # 这里的 err 的类型是 botocore.errorfactory.NoSuchKey 或 NoSuchBucket
                # 但是该类型是通过
                # type(exception_name, (ClientError,), {})    // botocore.errorfactory.py:83
                # 运行时构造的，目前的办法只能通过其基类 ClientError 来捕捉
                raise NoSuchKeyError(cluster, bucket, key)
            elif type(err).__name__ == 'NoSuchBucket':
                raise NoSuchBucketError(cluster, bucket)
            elif err.response['ResponseMetadata']['HTTPStatusCode'] == 403:
                raise AccessDeniedError(err)
            else:
                raise S3ClientError(err)

    def create_bucket(self, bucket):
        return self._s3_resource.create_bucket(Bucket=bucket)

    def isdir(self, bucket, key):
        itr = self.list(bucket, key)
        try:
            next(itr)
            return True
        except StopIteration:
            return False

    def list(self, bucket, key, page_size=None):
        if key is None:
            key = ''
        elif key and not key.endswith('/'):
            key = key + '/'

        client = self._s3_resource.meta.client
        paginator = client.get_paginator('list_objects')
        paging_args = {
            'Bucket': bucket, 'Prefix': key, 'Delimiter': '/',
            'PaginationConfig': {'PageSize': page_size}
        }
        itr = paginator.paginate(**paging_args)

        for response_data in itr:
            common_prefixes = response_data.get('CommonPrefixes', [])
            contents = response_data.get('Contents', [])

            for common_prefix in common_prefixes:
                prefix_components = common_prefix['Prefix'].split('/')
                prefix = prefix_components[-2]
                yield prefix + '/'

            for content in contents:
                filename_components = content['Key'].split('/')
                filename = filename_components[-1]
                yield filename

    def get_file_iterator(self, bucket, key):
        client = self._s3_resource.meta.client
        path = 's3://{0}'.format(bucket)
        if key:
            path = path + '/' + key
        file_iterator = FileIterator(client, path)
        return file_iterator

    @profile('put')
    def put_with_info(self, cluster, bucket, key, body, **kwargs):
        if isinstance(body, (bytes, bytearray)):
            result, info = self.put_bytes(cluster, bucket, key, body, **kwargs)
        elif hasattr(body, 'read'):
            result, info = self.multipart_upload(
                cluster, bucket, key, body, **kwargs)
        else:
            raise TypeError(
                f'{type(self)} does not support content type {type(body)}')

        if kwargs.get('enable_etag', False):
            info['etag'] = result.e_tag.strip('"')

        return result, info

    def put_bytes(self, cluster, bucket, key, body, **kwargs):
        assert self._cluster == cluster
        enable_md5 = kwargs.get('enable_md5', False)
        info = {}
        try:
            obj = self._s3_resource.Object(bucket, key)
            obj.put(Body=body)
            if enable_md5:
                info['md5'] = hashlib.md5(body).hexdigest()
            return obj, info
        except BotoClientError as err:
            if err.response['ResponseMetadata']['HTTPStatusCode'] == 403:
                raise AccessDeniedError(err)
            else:
                raise S3ClientError(err)

    def multipart_upload(self, cluster, bucket, key, stream, chunk_size=1024 * 1024 * 1024 * 2, **kwargs):
        assert self._cluster == cluster
        info = {}
        obj = self._s3_resource.Object(bucket, key)
        multipart = obj.initiate_multipart_upload()
        part_id = 0
        parts = []
        total_size = 0

        enable_md5 = kwargs.get('enable_md5', False)
        if enable_md5:
            md5 = hashlib.md5()

        while True:
            chunk = stream.read(chunk_size)
            actual_size = len(chunk)
            if actual_size == 0:
                break
            part_id += 1
            total_size += actual_size
            part = multipart.Part(part_id)
            response = part.upload(Body=chunk)
            parts.append({
                "PartNumber": part_id,
                "ETag": response["ETag"]
            })
            if enable_md5:
                md5.update(chunk)

        part_info = {
            'Parts': parts
        }
        result = multipart.complete(MultipartUpload=part_info)
        if enable_md5:
            info['md5'] = md5.hexdigest()
        return result, info

    def contains(self, cluster, bucket, key):
        assert self._cluster == cluster
        try:
            self._s3_resource.Object(bucket, key).load()
            return True
        except BotoClientError as err:
            if err.response['ResponseMetadata']['HTTPStatusCode'] == 404:
                return False
            elif err.response['ResponseMetadata']['HTTPStatusCode'] == 403:
                raise AccessDeniedError(err)
            else:
                raise S3ClientError(err)

    def delete(self, cluster, bucket, key, **kwargs):
        assert self._cluster == cluster
        try:
            return self._s3_resource.Object(bucket, key).delete()
        except BotoClientError as err:
            if type(err).__name__ == 'NoSuchKey':
                raise NoSuchKeyError(cluster, bucket, key)
            elif type(err).__name__ == 'NoSuchBucket':
                raise NoSuchBucketError(cluster, bucket)
            elif err.response['ResponseMetadata']['HTTPStatusCode'] == 403:
                raise AccessDeniedError(err)
            else:
                raise S3ClientError(err)

    def generate_presigned_url(self, cluster, bucket, key, client_method, expires_in):
        assert self._cluster == cluster
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.generate_presigned_url
        return self._s3_resource.meta.client.generate_presigned_url(
            client_method,
            {'Bucket': bucket, 'Key': key},
            expires_in
        )

    def generate_presigned_post(self, cluster, bucket, key, fields=None, conditions=None, expires_in=3600):
        assert self._cluster == cluster
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.generate_presigned_post
        return self._s3_resource.meta.client.generate_presigned_post(
            bucket, key, fields, conditions, expires_in
        )
