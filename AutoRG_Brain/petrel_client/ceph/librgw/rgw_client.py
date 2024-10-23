import logging

from petrel_client.ceph.ceph import Ceph
from petrel_client.common.io_profile import profile
from petrel_client.common.exception import ObjectNotFoundError
from petrel_client.ceph.librgw import rgw


LOG = logging.getLogger(__name__)


class RGWClient(Ceph):

    kb = 1024
    mb = 1024 * kb

    def __init__(self, cluster, conf, *args, **kwargs):
        LOG.debug('init RGWClient(%s)', cluster)
        super(RGWClient, self).__init__(cluster, conf, *args, **kwargs)
        conn_args = {
            'conf': conf['conf'],
            'keyring': conf['keyring'],
            'name': conf['name'],
            'cluster': conf['cluster'],
        }
        uid = conf.get('uid', 'user_id')
        self.bucket_fs = {}
        self._init_librgw(uid, conf['access_key'],
                          conf['secret_key'], conn_args)

    def _init_librgw(self, uid=None, key=None, secret=None, connection_kwargs=None):
        try:
            self.client = rgw.LibRGWFS(uid, key, secret, **connection_kwargs)
            self.root_fs = self.client.mount()
            LOG.debug("The connection bulid successfully.")
        except Exception as e:
            LOG.error("The input parameters is invalid. %s", e)
            raise Exception("The input parameters is invalid.", e)

    @profile('get')
    def get(self, cluster, bucket, key, file_size=4 * mb, **kwargs):
        try:
            # destination_base_uri = S3Uri(filename)
            # bucket = destination_base_uri.bucket()
            # key = destination_base_uri.object()
            bucket_fs = self.bucket_fs.get(bucket)
            if not bucket_fs:
                bucket_fs = self.client.opendir(self.root_fs, bucket)
                self.bucket_fs[bucket] = bucket_fs
            file_fs = self.client.open(bucket_fs, key)
            value = self.client.read(file_fs, 0, file_size)
            self.client.close(file_fs)
            self.client.close(bucket_fs)
            # log.debug('filename is: {}'.format(key))
            # log.debug('value size is: {} kB'.format(len(value) / self.kb))
            return value
        except rgw.ObjectNotFound as err:
            raise ObjectNotFoundError(err)

    def put(self, cluster, bucket, key, body, **kwargs):
        # destination_base_uri = S3Uri(filename)
        # bucket = destination_base_uri.bucket()
        # key = destination_base_uri.object()
        bucket_fs = self.client.opendir(self.root_fs, bucket)
        try:
            file_fs = self.client.create(bucket_fs, key)
        except rgw.ObjectExists:
            file_fs = self.client.open(bucket_fs, key)
        self.client.write(file_fs, 0, body)
        self.client.close(file_fs)
        self.client.close(bucket_fs)
        # log.debug('filename is: {}'.format(key))
        # log.debug('value size is: {} kB'.format(len(body) / self.kb))
        return True
