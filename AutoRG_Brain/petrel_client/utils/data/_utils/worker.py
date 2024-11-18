from petrel_client.utils.profile import profileit, wrap_with_stat_qsize, WORKER_LOOP_PROFILE_COUNT
from torch.utils.data._utils.worker import (
    torch, random, queue, ExceptionWrapper, ManagerWatchdog, WorkerInfo,
    _IterableDatasetStopIteration, signal_handling, MP_STATUS_CHECK_INTERVAL)


class _ResumeIteration(object):
    r"""Dummy class used to resume the fetching when worker reuse is enabled"""
    pass


def _worker_loop(dataset_kind, dataset, index_queue, data_queue, done_event,
                 auto_collation, collate_fn, drop_last, seed, init_fn, worker_id,
                 num_workers, persistent_workers):
    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    index_queue_get = wrap_with_stat_qsize(
        index_queue, index_queue.get, '_worker_loop.index_queue.qsize', count=WORKER_LOOP_PROFILE_COUNT)
    data_queue_put = wrap_with_stat_qsize(
        data_queue, data_queue.put, '_worker_loop.data_queue.qsize', count=WORKER_LOOP_PROFILE_COUNT)
    index_queue.get = index_queue_get
    data_queue.put = data_queue_put

    try:
        # Initialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
        # module's handlers are executed after Python returns from C low-level
        # handlers, likely when the same fatal signal had already happened
        # again.
        # https://docs.python.org/3/library/signal.html#execution-of-python-signal-handlers
        signal_handling._set_worker_signal_handlers()

        torch.set_num_threads(1)
        random.seed(seed)
        torch.manual_seed(seed)

        _worker_info = WorkerInfo(id=worker_id, num_workers=num_workers,
                                  seed=seed, dataset=dataset)

        from torch.utils.data._utils import worker as pt_worker
        pt_worker._worker_info = _worker_info

        from torch.utils.data import _DatasetKind

        init_exception = None

        try:
            if init_fn is not None:
                init_fn(worker_id)

            fetcher = _DatasetKind.create_fetcher(
                dataset_kind, dataset, auto_collation, collate_fn, drop_last)
        except Exception:
            init_exception = ExceptionWrapper(
                where="in DataLoader worker process {}".format(worker_id))

        # When using Iterable mode, some worker can exit earlier than others due
        # to the IterableDataset behaving differently for different workers.
        # When such things happen, an `_IterableDatasetStopIteration` object is
        # sent over to the main process with the ID of this worker, so that the
        # main process won't send more tasks to this worker, and will send
        # `None` to this worker to properly exit it.
        #
        # Note that we cannot set `done_event` from a worker as it is shared
        # among all processes. Instead, we set the `iteration_end` flag to
        # signify that the iterator is exhausted. When either `done_event` or
        # `iteration_end` is set, we skip all processing step and just wait for
        # `None`.
        iteration_end = False

        watchdog = ManagerWatchdog()
        cnt = 1
        brk = 0

        def loop():
            nonlocal iteration_end, init_exception, fetcher
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                return cnt
            if isinstance(r, _ResumeIteration):
                # Acknowledge the main process
                data_queue.put(r)
                iteration_end = False
                # Recreate the fetcher for worker-reuse policy
                fetcher = _DatasetKind.create_fetcher(
                    dataset_kind, dataset, auto_collation, collate_fn, drop_last)
                return cnt
            elif r is None:
                # Received the final signal
                assert done_event.is_set() or iteration_end
                return brk
            elif done_event.is_set() or iteration_end:
                # `done_event` is set. But I haven't received the final signal
                # (None) yet. I will keep continuing until get it, and skip the
                # processing steps.
                return cnt
            idx, index = r
            if init_exception is not None:
                data = init_exception
                init_exception = None
            else:
                try:
                    data = fetcher.fetch(index)
                except Exception as e:
                    if isinstance(e, StopIteration) and dataset_kind == _DatasetKind.Iterable:
                        data = _IterableDatasetStopIteration(worker_id)
                        # Set `iteration_end`
                        #   (1) to save future `next(...)` calls, and
                        #   (2) to avoid sending multiple `_IterableDatasetStopIteration`s.
                        iteration_end = True
                    else:
                        # It is important that we don't store exc_info in a variable.
                        # `ExceptionWrapper` does the correct thing.
                        # See NOTE [ Python Traceback Reference Cycle Problem ]
                        data = ExceptionWrapper(
                            where="in DataLoader worker process {}".format(worker_id))
            data_queue.put((idx, data))
            del data, idx, index, r  # save memory

        loop = profileit(loop, name='_worker_loop.loop',
                         count=WORKER_LOOP_PROFILE_COUNT)
        while watchdog.is_alive():
            if loop() == brk:
                break
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass
    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()
