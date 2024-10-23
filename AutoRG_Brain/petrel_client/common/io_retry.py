# -*- coding: utf-8 -*-

import functools
import logging
from humanize import ordinal

LOG = logging.getLogger(__name__)


def retry(op_name, exceptions=Exception, raises=(), tries=1):
    assert isinstance(op_name, str)

    def wrap(fn):
        @functools.wraps(fn)
        def new_fn(self, *args, **kwargs):
            return _retry(op_name, exceptions, raises, tries, fn, self, *args, **kwargs)
        return new_fn

    return wrap


def _retry(op_name, exceptions, raises, tries, fn, client, *args, **kwargs):
    uri, retry_max = args[0], tries
    for count in range(1, retry_max + 1):
        try:
            return fn(client, *args, **kwargs)
        except raises:
            raise
        except exceptions as err:
            if count < retry_max:
                LOG.debug('Exception occurred in the %s retry of %s operation on (%s): %s',
                          ordinal(count), op_name, uri, err)
                continue
            if retry_max > 1:
                LOG.error('%s operation (%s) has tried %s times and failed: %s',
                          op_name.capitalize(), uri, retry_max, err)
            raise
