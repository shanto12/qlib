# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
About the configs
=================
The config will be based on _default_config.
Two modes are supported
- client
- server
"""
from __future__ import annotations
import os
import re
import copy
import logging
import platform
import multiprocessing
import warnings
from pathlib import Path
from typing import Callable, Optional, Union
from typing import TYPE_CHECKING
from qlib.constant import REG_CN, REG_US, REG_TW

if TYPE_CHECKING:
    from qlib.utils.time import Freq

from pydantic_settings import BaseSettings, SettingsConfigDict

# NumPy 2.0+ and gym compatibility guards
try:
    import numpy as np
    NUMPY_VERSION = tuple(map(int, np.__version__.split('.')[:2]))
    if NUMPY_VERSION >= (2, 0):
        warnings.warn(
            f"NumPy {np.__version__} (2.0+) detected. Some APIs may have changed. "
            "If you encounter issues, consider pinning to numpy<2.0 or updating dependent code.",
            UserWarning
        )
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    NUMPY_VERSION = (0, 0)
    warnings.warn("NumPy not found. Some functionality may be limited.", UserWarning)

try:
    import gym
    GYM_AVAILABLE = True
    GYM_VERSION = getattr(gym, '__version__', '0.0.0')
except ImportError:
    GYM_AVAILABLE = False
    GYM_VERSION = '0.0.0'
    warnings.warn(
        "gym not found. RL-related functionality will be unavailable. "
        "Install with: pip install 'gym<0.26' or 'gymnasium' for newer API.",
        UserWarning
    )


class MLflowSettings(BaseSettings):
    uri: str = "file:" + str(Path(os.getcwd()).resolve() / "mlruns")
    default_exp_name: str = "Experiment"


class QSettings(BaseSettings):
    """
    Qlib's settings.
    It tries to provide a default settings for most of Qlib's components.
    But it would be a long journey to provide a comprehensive settings for all of Qlib's components.
    Here is some design guidelines:
    - The priority of settings is
        - Actively passed-in settings, like `qlib.init(provider_uri=...)`
        - The default settings
            - QSettings tries to provide default settings for most of Qlib's components.
    """

    mlflow: MLflowSettings = MLflowSettings()
    provider_uri: str = "~/.qlib/qlib_data/cn_data"
    model_config = SettingsConfigDict(
        env_prefix="QLIB_",
        env_nested_delimiter="_",
    )


QSETTINGS = QSettings()


class Config:
    def __init__(self, default_conf):
        self.__dict__["_default_config"] = copy.deepcopy(default_conf)  # avoiding conflicts with __getattr__
        self.reset()

    def __getitem__(self, key):
        return self.__dict__["_config"][key]

    def __getattr__(self, attr):
        if attr in self.__dict__["_config"]:
            return self.__dict__["_config"][attr]
        raise AttributeError(f"No such `{attr}` in self._config")

    def __setitem__(self, key, value):
        self.__dict__["_config"][key] = value

    def __contains__(self, item):
        return item in self.__dict__["_config"]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return str(self.__dict__["_config"])

    def __repr__(self):
        return str(self.__dict__["_config"])

    def update(self, *args, **kwargs):
        self.__dict__["_config"].update(*args, **kwargs)

    def get(self, key, default=None):
        return self.__dict__["_config"].get(key, default)

    def set_mode(self, mode):
        # raise ValueError("This func is not supported....")
        pass

    def set_region(self, region):
        self["region"] = region
        # update auto_path
        self["auto_path"] = get_auto_path_config(region)
        # update freq
        if "freq" not in self or self["freq"] == "day":
            self["freq"] = "day"  # keep the default freq
        elif self["freq"] == "1min":
            if region == REG_TW:
                self["freq"] = "1min"  # TW market supports 1min
            else:
                logger = logging.getLogger(__name__)
                logger.warning(f"freq 1min is not supported in {region} market, use day freq instead")
                self["freq"] = "day"

    def resolve_path(self):
        # resolve path
        for p in ["mount_path", "provider_uri"]:
            if p in self:
                self[p] = str(Path(os.path.expanduser(self[p])).resolve())

    def reset(self):
        self.__dict__["_config"] = copy.deepcopy(self.__dict__["_default_config"])


def get_auto_path_config(region):
    if region == REG_CN:
        return {
            "market_close": "16:00",
            "market_open": "09:30",
            "dpm_config": {
                "1d": {
                    "0900": [
                        [
                            "0930",
                        ],
                    ],
                },
            },
        }
    elif region == REG_US:
        return {
            "market_close": "16:00",
            "market_open": "09:30",
            "dpm_config": {
                "1d": {
                    "0900": [
                        [
                            "0930",
                        ],
                    ],
                },
            },
        }
    elif region == REG_TW:
        return {
            "market_close": "13:30",
            "market_open": "09:00",
            "dpm_config": {
                "1d": {
                    "0830": [
                        [
                            "0900",
                        ],
                    ],
                },
            },
        }
    else:
        raise ValueError(f"Unknown region {region}")



def default_kernel_config(freq: str):
    """
    The default kernel config for different frequencies.
    """
    if freq == "1min":
        return 20
    else:
        return multiprocessing.cpu_count() - 1


# default config
_default_config = {
    "exp_manager": {
        "class": "MLflowExpManager",
        "module_path": "qlib.workflow.expm",
        "kwargs": {"uri": QSETTINGS.mlflow.uri, "default_exp_name": QSETTINGS.mlflow.default_exp_name},
    },
    "logging": {
        "level": logging.INFO,
    },
    "region": REG_CN,
    "kernels": default_kernel_config,
}


def can_use_cache():
    """
    Check whether the redis server is available. So that we can skip the distributed
    execution.
    """
    res = True
    try:
        import redis
    except ImportError:
        return False

    try:
        r = redis.StrictRedis(host=C["redis_host"], port=C["redis_port"])
        r.ping()
    except (redis.exceptions.ConnectionError, KeyError):
        res = False
    return res


class QlibConfig(Config):
    _registered = False

    def set(self, default_conf="client", **kwargs):
        """
        Setup the qlib
        Parameters
        ----------
        default_conf : str
            The default config (from code) for the qlib
        """
        logger = logging.getLogger(__name__)
        if isinstance(default_conf, str):
            if default_conf == "server":
                # FIXME: this code is deprecated...
                # config for server (only consider 1d data, exclude code in the dataset, and so on.)
                default_conf = copy.deepcopy(_default_config)
                # For the server, we don't care about whether the cache is enabled. Because not all providers support it
                default_conf["expression_cache"] = None
                default_conf["dataset_cache"] = None
            elif default_conf == "client":
                # FIXME: this code is deprecated...
                default_conf = copy.deepcopy(_default_config)
            else:
                raise ValueError(f"Unknown default config: {default_conf}")
        else:
            default_conf = copy.deepcopy(default_conf)
        self.__dict__["_default_config"] = default_conf
        self.reset()
        # setup logging
        logging_conf = default_conf.get("logging", {})
        logging.basicConfig(**logging_conf)
        logger.info(f"default_conf: {default_conf}.")
        self.set_mode(default_conf)
        self.set_region(kwargs.get("region", self["region"] if "region" in self else REG_CN))
        for k, v in kwargs.items():
            if k not in self:
                logger.warning("Unrecognized config %s" % k)
            self[k] = v
        self.resolve_path()
        if not (self["expression_cache"] is None and self["dataset_cache"] is None):
            # check redis
            if not can_use_cache():
                log_str = ""
                # check expression cache
                if self.is_depend_redis(self["expression_cache"]):
                    log_str += self["expression_cache"]
                    self["expression_cache"] = None
                # check dataset cache
                if self.is_depend_redis(self["dataset_cache"]):
                    log_str += f" and {self['dataset_cache']}" if log_str else self["dataset_cache"]
                    self["dataset_cache"] = None
                if log_str:
                    logger.warning(
                        f"redis connection failed(host={self['redis_host']} port={self['redis_port']}), "
                        f"{log_str} will not be used!"
                    )

    def register(self):
        from .utils import init_instance_by_config  # pylint: disable=C0415
        from .data.ops import register_all_ops  # pylint: disable=C0415
        from .data.data import register_all_wrappers  # pylint: disable=C0415
        from .workflow import R, QlibRecorder  # pylint: disable=C0415
        from .workflow.utils import experiment_exit_handler  # pylint: disable=C0415

        register_all_ops(self)
        register_all_wrappers(self)

        # set up QlibRecorder
        exp_manager = init_instance_by_config(self["exp_manager"])
        qr = QlibRecorder(exp_manager)
        R.register(qr)
        # clean up experiment when python program ends
        experiment_exit_handler()

        # Supporting user reset qlib version (useful when user want to connect to qlib server with old version)
        self.reset_qlib_version()
        self._registered = True

    def reset_qlib_version(self):
        import qlib  # pylint: disable=C0415

        reset_version = self.get("qlib_reset_version", None)
        if reset_version is not None:
            qlib.__version__ = reset_version
        else:
            qlib.__version__ = getattr(qlib, "__version__bak")
            # Due to a bug? that converting __version__ to _QlibConfig__version__bak
            # Using  __version__bak instead of __version__

    def get_kernels(self, freq: str):
        """get number of processors given frequency"""
        if isinstance(self["kernels"], Callable):
            return self["kernels"](freq)
        return self["kernels"]

    @property
    def registered(self):
        return self._registered

    @staticmethod
    def is_depend_redis(cache_config):
        """Check if the cache depends on redis"""
        if cache_config is None:
            return False
        if isinstance(cache_config, dict):
            if cache_config.get("class", "").endswith("DiskExpressionCache") or cache_config.get(
                "class", ""
            ).endswith("DiskDatasetCache"):
                return False
        return True


# global config
C = QlibConfig(_default_config)
