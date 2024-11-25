from __future__ import annotations

import os
from configparser import ConfigParser
from itertools import chain
from pathlib import Path

# Global _CONFIG used by other packages
_CONFIG = {}


def reload_config(package=None):
    """Reloads the configuration from the config file.

    Used for forcibly reloading the configuration from the config file, particularly useful for pytesting mock imports.
    """
    global _CONFIG
    _load_config(package)

    for key, value in _CONFIG["environment"].items():
        os.environ[key.upper()] = value

    return _CONFIG


def get_config(package=None):
    global _CONFIG

    if not _CONFIG:
        _load_config(package)

    return _CONFIG


def _gather_config_files(package: str | os.PathLike[str] | Path | None = None):
    conf_files = []
    _config = Path(__file__).parent.joinpath("etc").joinpath("roocs.ini")

    # add default config file
    # FIXME: we should be using importlib.resources to get the default config file
    if not _config.is_file():
        print(f"[WARN] Cannot load default config file from: {_config.as_posix()}")
    else:
        conf_files.append(_config)
    if package:
        pkg_config = Path(package).parent.joinpath("etc").joinpath("roocs.ini")
        if pkg_config.is_file():
            conf_files.append(pkg_config)

    # add system config /etc/roocs.ini
    sys_config = Path(Path(os.sep, "etc", "roocs.ini")).absolute()
    if sys_config.is_file():
        conf_files.append(sys_config)

    # add custom config from environment variable
    roocs_config = "ROOCS_CONFIG"
    if roocs_config in os.environ:
        conf_files.extend([Path(p) for p in os.environ[roocs_config].split(":")])

    return conf_files


def _to_list(i):
    return i.split()


def _to_dict(i):
    if not i.strip():
        return {}
    return dict([_.split(":") for _ in i.strip().split("\n")])


def _to_int(i):
    return int(i)


def _to_float(i):
    return float(i)


def _to_boolean(i):
    if i == "True":
        return True
    elif i == "False":
        return False
    else:
        raise ValueError(
            f"{i} is not valid for a boolean field - use 'True' or 'False'"
        )


def _chain_config_types(conf, keys):
    return chain(
        *[
            conf.get("config_data_types", key).split()
            for key in keys
            if conf.has_option("config_data_types", key)
        ]
    )


def _get_mappers(conf):
    mappers = {}

    for key in _chain_config_types(conf, ["lists", "extra_lists"]):
        mappers[key] = _to_list

    for key in _chain_config_types(conf, ["dicts", "extra_dicts"]):
        mappers[key] = _to_dict

    for key in _chain_config_types(conf, ["ints", "extra_ints"]):
        mappers[key] = _to_int

    for key in _chain_config_types(conf, ["floats", "extra_floats"]):
        mappers[key] = _to_float

    for key in _chain_config_types(conf, ["boolean", "extra_booleans"]):
        mappers[key] = _to_boolean

    return mappers


def _load_config(package=None):
    global _CONFIG

    conf_files = _gather_config_files(package)
    conf = ConfigParser()

    conf.read(conf_files)
    config = {}

    mappers = _get_mappers(conf)

    for section in conf.sections():
        config.setdefault(section, {})

        for key in conf.options(section):
            value = conf.get(section, key)

            if key in mappers:
                value = mappers[key](value)

            config[section][key] = value

    _post_process(config)

    _CONFIG = config


def _post_process(config):
    """
    Post-processes the contents of the config file to modify sections based on
    certain rules.
    Returns None.
    """
    for name in [n for n in config.keys() if n.startswith("project:")]:
        _modify_fixed_path_mappings(config, name)


def _modify_fixed_path_mappings(config, name):
    """
    Expands the contents of `fixed_path_mappings` based on other fixed path modifiers`.
    Returns None - changes contents in place.
    """
    d = config[name]

    fp_mappings = "fixed_path_mappings"
    fp_modifiers = "fixed_path_modifiers"

    if fp_mappings not in d or fp_modifiers not in d:
        return

    mappings = d[fp_mappings].copy()

    for modifier in d[fp_modifiers]:
        items = d[fp_modifiers][modifier].split()
        mappings = _expand_mappings(mappings, modifier, items)

    d[fp_mappings] = mappings.copy()


def _expand_mappings(mappings, modifier, items):
    """
    Expands mappings by replacing modifier with list of items
    in each case.
    """
    result = {}

    for key, value in mappings.items():
        lookup = "{" + modifier + "}"

        if lookup in key or lookup in value:
            for item in items:
                result[key.replace(lookup, item)] = value.replace(lookup, item)
        else:
            result[key] = value

    return result
