# -*- coding: utf-8 -*-

import functools
import logging
import threading
import weakref
import environs
from time import time
from collections import defaultdict
import io

from petrel_client.common import mem_trace
from petrel_client.common.exception import ObjectNotFoundError

LOG = logging.getLogger(__name__)
ENV = environs.Env()


class StatItem(object):
    __slots__ = ['op_name', 'total_io', 'total_hit',
                 'total_time', 'total_error', 'total_miss',
                 'error_count', 'total_byte'
                 ]

    def __init__(self, op_name):
        self.op_name = op_name
        self.reset()

    def reset(self):
        self.total_io = 0
        self.total_hit = 0
        self.total_time = 0.0
        self.total_error = 0
        self.total_miss = 0
        self.total_byte = 0
        self.error_count = defaultdict(lambda: 0)

    @property
    def time_avg(self):
        return self.total_time / self.total_io if self.total_io else .0

    @property
    def hit_ratio(self):
        return 1.0 * self.total_hit / self.total_io if self.total_io else .0

    @property
    def speed(self):
        return 1.0 * self.total_byte / self.total_time if self.total_time else .0

    def stat_io(self, callback=None):
        stat_info = f'{self.op_name} [total: {self.total_io}' \
            f', hit: {self.total_hit}' \
            f', miss: {self.total_miss}' \
            f', error: {self.total_error}' \
            f', time: {self.total_time:.6} s' \
            f', time_avg: {self.time_avg:.6} s' \
            f', hit ratio: {self.hit_ratio:.2%}' \
            f', bytes: {_sizeof_fmt(self.total_byte)}' \
            f', speed: {_sizeof_fmt(self.speed,suffix="B/s")}' \
            f']'

        if self.error_count:
            items = ["{}: {}".format(k, v)
                     for (k, v) in self.error_count.items()]
            stat_info = f'{stat_info}, error_count: [{", ".join(items)}]'

        if callback:
            callback(stat_info)
        else:
            LOG.info(stat_info)
        self.reset()


class StatItemDict(dict):
    def __missing__(self, key):
        item = self[key] = StatItem(key)
        return item

    def stat_io(self, callback=None):
        for item in self.values():
            item.stat_io(callback)


class ClientStat(object):
    def __init__(self, client_id, name):
        self.client_id = client_id
        self.name = name
        self.stat_item_dict = StatItemDict()
        profiler = Profiler.get()
        self.profiler = profiler
        profiler.register(self)

    def __getitem__(self, op_name):
        return self.stat_item_dict[op_name]

    # 若使用 multiprocessing-logging，进程退出时候调用 __del__ 还存在问题
    # def __del__(self):
    #     # 这里有可能是再 python 将要退出的时候触发，此时file log已经不存在，会发生异常
    #     try:
    #         self.profiler.unregister(self)
    #         if self.total_io:
    #             self.stat_io()
    #     except Exception:
    #         pass

    @property
    def total_io(self):
        return sum([item.total_io for item in self.stat_item_dict.values()])

    @property
    def get_hit(self):
        return sum([item.total_hit for item in self.stat_item_dict.values() if item.op_name == 'get'])

    def stat_io(self, callback=None):
        stat_item_info_list = []

        def cb(info):
            stat_item_info_list.append(info)

        for stat_item in self.stat_item_dict.values():
            stat_item.stat_io(cb)

        if stat_item_info_list:
            stat_itme_info = ', '.join(stat_item_info_list)
        else:
            stat_itme_info = 'No IO operations'

        stat_info = '{}: {}'.format(self.name, stat_itme_info)
        if callback:
            callback(stat_info)
        else:
            LOG.info(stat_info)


def profile(op_name):
    assert isinstance(op_name, str)

    def wrap(fn):
        @functools.wraps(fn)
        def new_fn(self, *args, **kwargs):
            return _profile(op_name, fn, self, *args, **kwargs)
        return new_fn

    return wrap


def _profile(op_name, fn, client, *args, **kwargs):
    stat: StatItem = client.client_stat[op_name]
    start = time()
    try:
        ret = fn(client, *args, **kwargs)
        if isinstance(ret, (tuple, list)):
            content = ret[0]
        else:
            content = ret

        if isinstance(content, bytes):
            stat.total_byte += len(content)
        elif isinstance(content, int):
            stat.total_byte += content
        elif hasattr(content, 'content_length'):
            stat.total_byte += content.content_length
        elif op_name == 'get' and content is None:
            raise ObjectNotFoundError()

        stat.total_hit += 1
        return ret

    except ObjectNotFoundError:
        stat.total_miss += 1
        raise

    except Exception as e:
        stat.total_error += 1
        err_name = e.__class__.__name__
        stat.error_count[err_name] += 1
        raise

    finally:
        end = time()
        stat.total_time += (end - start)
        stat.total_io += 1
        client.client_stat.profiler.inc_op_count()


class Profiler(object):
    thread_local = threading.local()
    default_conf = None

    @staticmethod
    def set_default_conf(conf):
        Profiler.default_conf = conf

    @staticmethod
    def get():
        profiler = getattr(Profiler.thread_local, 'profiler', None)
        if not profiler:
            profiler = Profiler(Profiler.default_conf)
            setattr(Profiler.thread_local, 'profiler', profiler)
        return profiler

    def __init__(self, conf, *args, **kwargs):
        assert conf is not None
        self.stat_dict = weakref.WeakValueDictionary()
        self.op_count = 0
        self.count_disp = ENV.int(
            'count_disp', None) or conf.get_int('count_disp')

        self.enable_mem_trace = conf.get_boolean('enable_mem_trace')
        if self.enable_mem_trace:
            mem_trace.start()

    def register(self, client_stat: ClientStat):
        client_id = client_stat.client_id
        self.stat_dict[client_id] = client_stat

    def unregister(self, client_stat: ClientStat):
        client_id = client_stat.client_id
        del self.stat_dict[client_id]

    def inc_op_count(self):
        self.op_count += 1
        if self.count_disp:
            if self.op_count >= self.count_disp:
                self.stat_io()
                self.op_count = 0

    @staticmethod
    def set_count_disp(count_disp):
        if count_disp < 0:
            LOG.error('count_disp must be a nonnegative integer, actual value: %s',
                      count_disp)
            return

        profiler = Profiler.get()
        profiler.count_disp = count_disp

    def stat_io(self):
        if LOG.isEnabledFor(logging.INFO):
            io_dict = {
                client_stat.name: client_stat.get_hit for client_stat in self.stat_dict.values()}
            total_io = sum(io_dict.values()) or 1
            percentage = [f'{client_name}: {1.0 * count / total_io :.2%}' for client_name,
                          count in io_dict.items()]

            for client_stat in self.stat_dict.values():
                client_stat.stat_io()
            LOG.info('IO Percentage: %s', ', '.join(percentage))
            if self.enable_mem_trace:
                snapshot = mem_trace.take_snapshot()
                buffer = io.StringIO()
                snapshot.display_top(buffer=buffer)
                LOG.info('Memory trace: \n%s', buffer.getvalue())

    def enable(self):
        raise NotImplementedError()

    def disable(self):
        raise NotImplementedError()


def _sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)
