from torch.utils.data._utils.pin_memory import (
    torch, queue, pin_memory, MP_STATUS_CHECK_INTERVAL, ExceptionWrapper)

from .worker import _ResumeIteration

from petrel_client.utils.profile import profileit, wrap_with_stat_qsize


def _pin_memory_loop(in_queue, out_queue, device_id, done_event):
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)

    torch.cuda.set_device(device_id)

    in_queue_get = wrap_with_stat_qsize(
        in_queue, in_queue.get, '_pin_memory_loop.in_queue.qsize:')
    out_queue_put = wrap_with_stat_qsize(
        out_queue, out_queue.put, '_pin_memory_loop.out_queue.qsize:')

    in_queue.get = in_queue_get
    out_queue.put = out_queue_put

    cnt = 1
    brk = 0

    def loop():
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            return cnt
        if not isinstance(r, _ResumeIteration):
            idx, data = r
            if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
                try:
                    data = pin_memory(data)
                except Exception:
                    data = ExceptionWrapper(
                        where="in pin memory thread for device {}".format(device_id))
                r = (idx, data)
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue
        del r  # save memory

    loop = profileit(loop, name='_pin_memory_loop.loop')
    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    while not done_event.is_set():
        if loop() == brk:
            break
