import datetime as dt
import atexit
import logging
import logging.config
import logging.handlers
from typing import Literal, override

logger = logging.getLogger(__name__)

logging_level = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def init_logger(level: logging_level):
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "[%(levelname)s|%(module)s|L%(lineno)d] %(asctime)s: %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
            "json": {
                "()": "core.logger.JSONFormatter",
                "fmt_keys": {
                    "level": "levelname",
                    "message": "message",
                    "timestamp": "timestamp",
                    "logger": "name",
                    "module": "module",
                    "function": "funcName",
                    "line": "lineno",
                    "thread_name": "threadName",
                },
            },
        },
        "handlers": {
            "stderr": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "json",
                "stream": "ext://sys.stderr",
            },
            "queue_handler": {
                "class": "logging.handlers.QueueHandler",
                "handlers": ["stderr"],
                "respect_handler_level": True,
            },
        },
        "loggers": {
            "root": {
                "level": level,
                "handlers": ["queue_handler"],
            }
        },
    }

    logging.config.dictConfig(logging_config)
    queue_handler: logging.handlers.QueueHandler | None = logging.getHandlerByName(
        "queue_handler"
    )  # type: ignore

    if queue_handler is not None:
        listener = queue_handler.listener

        if listener is not None:
            listener.start()
            atexit.register(listener.stop)


LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JSONFormatter(logging.Formatter):
    def __init__(
        self,
        *,
        fmt_keys: dict[str, str] | None = None,
    ):
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    @override
    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_dict(record)
        # return json.dumps(message, default=str)

        string = ""

        string += f"{message['timestamp']} {message['logger']} "

        if message["level"] == "DEBUG":
            string += "\033[1;90mDEBUG\033[0m    "
        elif message["level"] == "INFO":
            string += "\033[1;32mINFO\033[0m     "
        elif message["level"] == "WARNING":
            string += "\033[1;33mWARNING\033[0m  "
        elif message["level"] == "ERROR":
            string += "\033[1;31mERROR\033[0m    "
        elif message["level"] == "CRITICAL":
            string += "\033[1;7;31mCRITICAL\033[0m "

        string += f"{message['module']}:{message['function']}:{message['line']:<4}: {message['message']}"

        return string

    def _prepare_log_dict(self, record: logging.LogRecord):
        always_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        message = {
            key: msg_val
            if (msg_val := always_fields.pop(val, None)) is not None
            else getattr(record, val)
            for key, val in self.fmt_keys.items()
        }
        message.update(always_fields)

        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val

        return message


class NonErrorFilter(logging.Filter):
    @override
    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        return record.levelno <= logging.INFO
