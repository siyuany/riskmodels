# -*- encoding: utf-8 -*-
"""
logging.py: 日志模块
"""

import logging as _logging
import sys
from logging import DEBUG, INFO, WARN, ERROR, FATAL

_logging.basicConfig(format='[%(asctime)s] %(levelname)s - '
                     '%(name)s.%(funcName)s(%(lineno)s): %(message)s')


def set_verbosity(level):
    logger = _logging.getLogger()
    logger.setLevel(level)


def _log(level, msg, *args, **kwargs):
    # DO NOT USE THIS FUNCTION DIRECTLY!
    module_name = sys._getframe(2).f_globals['__name__']
    logger = _logging.getLogger(module_name)
    logger.log(level, msg, stacklevel=3, *args, **kwargs)


def debug(msg, *args, **kwargs):
    _log(DEBUG, msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    _log(INFO, msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    _log(WARN, msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    _log(ERROR, msg, *args, **kwargs)


def fatal(msg, *args, **kwargs):
    _log(FATAL, msg, *args, **kwargs)


__all__ = [
    'DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL', 'debug', 'info', 'warn', 'error',
    'fatal', 'set_verbosity'
]
