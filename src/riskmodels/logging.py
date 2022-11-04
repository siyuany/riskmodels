# -*- encoding: utf-8 -*-
"""
logging.py: 日志模块
"""

import logging as _logging
import sys
from logging import DEBUG, INFO, WARN, ERROR, FATAL

if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    pass
elif sys.version_info[0] == 3 and sys.version_info[1] in [6, 7]:
    import io
    import os
    import traceback

    def findCaller(self, stack_info=False, stacklevel=1):
        """
        Find the stack frame of the caller so that we can note the source
        file name, line number and function name.
        """
        f = _logging.currentframe()
        #On some versions of IronPython, currentframe() returns None if
        #IronPython isn't run with -X:Frames.
        if f is not None:
            f = f.f_back
        orig_f = f
        while f and stacklevel > 1:
            f = f.f_back
            stacklevel -= 1
        if not f:
            f = orig_f
        rv = "(unknown file)", 0, "(unknown function)", None
        while hasattr(f, "f_code"):
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            if filename == _logging._srcfile:
                f = f.f_back
                continue
            sinfo = None
            if stack_info:
                sio = io.StringIO()
                sio.write('Stack (most recent call last):\n')
                traceback.print_stack(f, file=sio)
                sinfo = sio.getvalue()
                if sinfo[-1] == '\n':
                    sinfo = sinfo[:-1]
                sio.close()
            rv = (co.co_filename, f.f_lineno, co.co_name, sinfo)
            break
        return rv

    def _log(self,
             level,
             msg,
             args,
             exc_info=None,
             extra=None,
             stack_info=False,
             stacklevel=1):
        """
        Low-level logging routine which creates a LogRecord and then calls
        all the handlers of this logger to handle the record.
        """
        sinfo = None
        if _logging._srcfile:
            #IronPython doesn't track Python frames, so findCaller raises an
            #exception on some versions of IronPython. We trap it here so that
            #IronPython can use logging.
            try:
                fn, lno, func, sinfo = self.findCaller(stack_info, stacklevel)
            except ValueError:  # pragma: no cover
                fn, lno, func = "(unknown file)", 0, "(unknown function)"
        else:  # pragma: no cover
            fn, lno, func = "(unknown file)", 0, "(unknown function)"
        if exc_info:
            if isinstance(exc_info, BaseException):
                exc_info = (type(exc_info), exc_info, exc_info.__traceback__)
            elif not isinstance(exc_info, tuple):
                exc_info = sys.exc_info()
        record = self.makeRecord(self.name, level, fn, lno, msg, args, exc_info,
                                 func, extra, sinfo)
        self.handle(record)

    _logging.Logger.findCaller = findCaller
    _logging.Logger._log = _log
else:
    raise RuntimeError(
        'Not support Python with version earlier than 3.6, get %s.%s' %
        (sys.version_info[0], sys.version_info[1]))

_logging.basicConfig(format='[%(asctime)s] %(levelname)s - '
                     '%(name)s.%(funcName)s(%(lineno)s): %(message)s',
                     level=ERROR)


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
