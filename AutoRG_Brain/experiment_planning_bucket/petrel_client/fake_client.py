from petrel_client.client_base import ClientBase
from petrel_client.common.io_profile import profile


class FakeClient(ClientBase):
    customized_get = None
    customized_put = None

    def __init__(self, client_type, conf, **kwargs):
        super(FakeClient, self).__init__(conf=conf, **kwargs)
        self.conf = conf
        self.type = client_type
        self.__enable_cache = conf.get_boolean(
            'enable_mc', False) or conf.get_boolean('enable_cache', False)

    @profile('get')
    def get(self, *args, **kwargs):
        if self.customized_get:
            return self.customized_get(*args, **kwargs)
        else:
            return b'data from FakeClient.'

    def get_with_info(self, *args, **kwargs):
        info = {}
        data = self.get(*args, **kwargs)
        return data, info

    @profile('put')
    def put(self, *args, **kwargs):
        if self.customized_put:
            return self.customized_put(*args, **kwargs)
        else:
            if self.type == 's3':
                body = args[3]
            else:
                body = args[1]
            return len(body)

    def put_with_info(self, *args, **kwargs):
        info = {}
        result = self.put(*args, **kwargs)
        return result, info

    def enable_cache(self):
        return self.__enable_cache
