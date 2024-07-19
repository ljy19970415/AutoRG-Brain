import re


_S3_ACCESSPOINT_TO_BUCKET_KEY_REGEX = re.compile(
    r'^(?P<bucket>arn:(aws).*:s3:[a-z\-0-9]+:[0-9]{12}:accesspoint[:/][^/]+)/?'
    r'(?P<key>.*)$'
)


def find_bucket_key(s3_path):
    """
    This is a helper function that given an s3 path such that the path is of
    the form: bucket/key
    It will return the bucket and the key represented by the s3 path
    """
    match = _S3_ACCESSPOINT_TO_BUCKET_KEY_REGEX.match(s3_path)
    if match:
        return match.group('bucket'), match.group('key')
    s3_components = s3_path.split('/', 1)
    bucket = s3_components[0]
    s3_key = ''
    if len(s3_components) > 1:
        s3_key = s3_components[1]
    return bucket, s3_key


class BucketLister(object):
    """List keys in a bucket."""

    def __init__(self, client):
        self._client = client

    def list_objects(self, bucket, prefix=None, page_size=1000, max_items=None):
        kwargs = {'Bucket': bucket,
                  'PaginationConfig': {'PageSize': page_size,
                                       'MaxItems ': max_items}}
        if prefix is not None:
            kwargs['Prefix'] = prefix

#       paginator = self._client.get_paginator('list_objects_v2')
        paginator = self._client.get_paginator('list_objects')
        pages = paginator.paginate(**kwargs)
        for page in pages:
            contents = page.get('Contents', [])
            for content in contents:
                source_path = bucket + '/' + content['Key']
                yield source_path, content


class FileGenerator(object):
    """
    This is a class the creates a generator to yield files based on information
    returned from the ``FileFormat`` class.  It is universal in the sense that
    it will handle s3 files, local files, local directories, and s3 objects
    under the same common prefix.  The generator yields corresponding
    ``FileInfo`` objects to send to a ``Comparator`` or ``S3Handler``.
    """

    def __init__(self, client, page_size=None):
        self._client = client
        self.page_size = page_size
        self.request_parameters = {}

    def __call__(self, path):

        file_iterator = self.list_objects(path)
        yield from file_iterator

    def list_objects(self, s3_path):
        """
        This function yields the appropriate object or objects under a
        common prefix depending if the operation is on objects under a
        common prefix.  It yields the file's source path, size, and last
        update.
        """
        if s3_path.startswith('s3://'):
            s3_path = s3_path[5:]
        bucket, prefix = find_bucket_key(s3_path)
        lister = BucketLister(self._client)
        for key in lister.list_objects(bucket=bucket, prefix=prefix,
                                       page_size=self.page_size):
            source_path, response_data = key
            if response_data['Size'] == 0 and source_path.endswith('/'):
                pass
            else:
                yield source_path, response_data


class FileIterator(object):

    def __init__(self, client, path, page_size=None):
        self._client = client
        self.path = path
        self.page_size = page_size
        self.request_parameters = {}

    def __iter__(self):
        generator = FileGenerator(self._client, self.page_size)
        return generator(self.path)
