import cProfile
import pstats
import io
import functools
import os
from collections import defaultdict
from distutils.util import strtobool

PETREL_PROFILE_ENV = os.getenv('PETREL_PROFILE', 'False')
try:
    ENABLE_PROFILE = strtobool(PETREL_PROFILE_ENV)
except ValueError:
    raise ValueError(
        f'invalid value of environment variable PETREL_PROFILE: {PETREL_PROFILE_ENV}')

PROFILE_COUNT_ENV = os.getenv('PETREL_PROFILE_COUNT', 1000)
try:

    PROFILE_COUNT = int(PROFILE_COUNT_ENV)
except ValueError:
    raise ValueError(
        f'invalid value of environment variable PETREL_PROFILE_COUNT: {PROFILE_COUNT_ENV}')

WORKER_LOOP_PROFILE_COUNT_ENV = os.getenv(
    'PETREL_WORKER_LOOP_PROFILE_COUNT', 250)
try:

    WORKER_LOOP_PROFILE_COUNT = int(WORKER_LOOP_PROFILE_COUNT_ENV)
except ValueError:
    raise ValueError(
        f'invalid value of environment variable PETREL_WORKER_LOOP_PROFILE_COUNT: {WORKER_LOOP_PROFILE_COUNT_ENV}')


def print_stats(prof, name, sortby='cumulative'):
    s = io.StringIO()
    if name:
        s.write(f'\nProfile of function {name}:\n')
    s.write(f'pid: {os.getpid()}\n')
    ps = pstats.Stats(prof, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


def profile_helper(func, name, count):
    if not ENABLE_PROFILE:
        return func

    prof = cProfile.Profile()
    call_count = 0

    if not name:
        try:
            name = func.__name__
        except AttributeError:
            pass

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal prof
        nonlocal call_count
        try:
            return prof.runcall(func, *args, **kwargs)
        finally:
            call_count += 1
            if call_count == count:
                print_stats(prof, name)
                call_count = 0
                prof = cProfile.Profile()

    return wrapper


def profileit(*args, name=None, count=PROFILE_COUNT):
    if args:
        assert len(args) == 1 and callable(args[0])
        return profile_helper(args[0], name, count)
    else:
        return functools.partial(profileit, name=name, count=count)


def wrap_with_stat_qsize(queue, cb, name, count=PROFILE_COUNT):
    if not ENABLE_PROFILE:
        return cb

    cb_count = 0
    qsize_dict = defaultdict(lambda: 0)
    qsize_list = []

    @functools.wraps(cb)
    def wrapper(*args, **kwargs):
        nonlocal cb_count
        cb_count += 1
        qsize = queue.qsize()
        qsize_dict[qsize] += 1
        qsize_list.append(qsize)
        try:
            return cb(*args, **kwargs)
        finally:
            if cb_count == count:
                print('pid', os.getpid(), name, qsize_dict, '\n', qsize_list)
                cb_count = 0
                qsize_dict.clear()
                qsize_list.clear()

    return wrapper
