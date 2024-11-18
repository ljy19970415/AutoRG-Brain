
import threading
import logging
from logging.handlers import RotatingFileHandler
import coloredlogs
import os
import socket

from petrel_client.version import version

# Level       Numeric value
#
# CRITICAL            50
# ERROR               40
# WARNING             30
# INFO                20
# DEBUG               10
# NOTSET               0

# https://docs.python.org/2.7/library/logging.html#logrecord-attributes
BASE_FORMAT = '%(asctime)s %(levelname).3s [%(processName)-11s] [%(threadName)-10s] - %(message)s [P:%(process)d T:%(thread)d F:%(filename)s:%(lineno)d]'
base_formatter = logging.Formatter(BASE_FORMAT)

log_config = {}
LOG = logging.getLogger('petrel_client')

LOG.propagate = False
LOG.setLevel(logging.DEBUG)

coloredlogs.install(level='DEBUG', logger=LOG,
                    milliseconds=True, fmt=BASE_FORMAT)
console_handler = LOG.handlers[0]

lock = threading.RLock()
log_config = {
    'have_initiated': False
}


def get_log_file_name():
    slurm_procid = os.environ.get('SLURM_PROCID', None)
    if slurm_procid is not None:
        file_name = f'slurm_procid_{slurm_procid}.log'
    else:
        hostname = socket.gethostname()
        pid = os.getpid()
        file_name = f'{hostname}_pid_{pid}.log'

    return file_name


def init_log(conf):
    with lock:
        if log_config['have_initiated']:
            LOG.debug('log initiated, skip')
            return
        else:
            log_config['have_initiated'] = True

    log_file_path = conf.get('log_file_path', None)
    if log_file_path:
        if not os.path.exists(log_file_path):
            # exist_ok = True : avoid FileExistsError when multiple
            # processes are trying to create the same log_file_path
            os.makedirs(log_file_path, exist_ok=True)

        file_log_level = conf.get_log_level('file_log_level')
        file_log_max_bytes = conf.get_int('file_log_max_bytes')
        file_log_backup_count = conf.get_int('file_log_backup_count')

        file_handler = RotatingFileHandler(
            filename=os.path.join(log_file_path, get_log_file_name()),
            maxBytes=file_log_max_bytes,
            backupCount=file_log_backup_count)
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(base_formatter)
        LOG.addHandler(file_handler)

    if conf.has_option('console_log_level'):
        console_log_level = conf.get_log_level('console_log_level')
        console_handler.setLevel(console_log_level)

    if log_file_path:
        from multiprocessing_logging import install_mp_handler
        # install_mp_handler should be invoked after log configuration
        install_mp_handler(LOG)

    LOG.debug('init log, SDK version %s', version)
