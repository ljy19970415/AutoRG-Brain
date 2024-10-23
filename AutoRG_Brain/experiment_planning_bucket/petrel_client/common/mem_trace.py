import linecache
import tracemalloc

from tracemalloc import Snapshot


def format_size(size):
    return tracemalloc._format_size(size, False)


def display_top(snapshot: Snapshot, limit=10, buffer=None, key_type='lineno'):
    if buffer:
        def write(msg):
            buffer.write(msg)
            buffer.write('\n')
    else:
        def write(msg):
            print(msg)

    stats = snapshot.statistics(key_type)

    for index, stat in enumerate(stats[:limit], 1):
        frame = stat.traceback[0]
        line = linecache.getline(frame.filename, frame.lineno).strip()
        msg = f'#{index}:\t{frame.filename}:{frame.lineno}: {stat.count} blocks, {format_size(stat.size)}\n\t{line}'
        write(msg)

    other = stats[limit:]
    if other:
        other_size = sum(stat.size for stat in other)
        other_blocks = sum(stat.count for stat in other)
        write(
            f'Other:\t{len(other)} items, {other_blocks} blocks, {format_size(other_size)}')

    total_size = sum(stat.size for stat in stats)
    total_blocks = sum(stat.count for stat in stats)
    write(
        f'Total:\t{len(stats)} items, {total_blocks} blocks, {format_size(total_size)}')


def start():
    tracemalloc.start()


def stop():
    tracemalloc.stop()


def take_snapshot():
    return tracemalloc.take_snapshot()


def filter_traces(snapshot, pattern):
    return snapshot.filter_traces((
        tracemalloc.Filter(True, pattern),
    ))


Snapshot.display_top = display_top
Snapshot.filter_traces = filter_traces
