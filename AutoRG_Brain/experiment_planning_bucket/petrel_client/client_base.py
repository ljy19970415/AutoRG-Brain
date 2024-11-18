import logging

from petrel_client.common.io_profile import ClientStat

LOG = logging.getLogger(__name__)


class ClientBase(object):

    def __init__(self, *args, **kwargs):
        cls_name = type(self).__name__
        name = kwargs.get('name', None)
        name = '{}({}id: {})'.format(
            cls_name,
            '{}, '.format(name) if name else '',
            id(self))

        self.client_stat = ClientStat(id(self), name)

        LOG.debug('create %s', name)
