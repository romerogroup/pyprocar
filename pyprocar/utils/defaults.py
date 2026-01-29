from __future__ import annotations

import configparser
import inspect
import os
from abc import ABCMeta
from collections.abc import Iterator, Mapping
from typing import Any, override

equivalents = {
    "cmap": "color_map",
    "cmaps": "color_map",
    "ylim": "elimit",
    "ylimit": "elimit",
    "efermi": "fermi",
    "mask": "projection_mask",
    "marker_size": "markersize",
    "colors": "color",
    "opacities": "opacity",
    "linewidths": "linewidth",
    "labels": "label",
}


class Settings:
    """Settings manager that loads from INI config files."""

    __metaclass__: type[ABCMeta] = ABCMeta
    config: dict[str, Any]

    def __init__(
        self,
        filename: str | None = None,
        config: Mapping[str, str] | None = None,
    ) -> None:
        self.config = {}
        if config is None and filename is not None:
            parser = configparser.ConfigParser()
            parser.read(filename)
            for item in parser.sections():
                sub = Settings(config=parser[item])
                self.__setattr__(item, sub)

        elif config is not None:
            for item in config:
                if "," in config[item]:
                    attr: Any = config[item].split(",")
                    attr = [type_convert(x) for x in attr]
                else:
                    attr = type_convert(config[item])
                self.__setattr__(item, attr)
        if config is not None:
            self.check_equivalents(config)

    def modify(self, changes: dict[str, Any]) -> None:
        """Maybe needs modification to specify section to change."""
        changes = {item: changes[item] for item in changes}
        for item in changes:
            if item in self.config:
                self.__setattr__(item, changes[item])

    def check_equivalents(self, config: Mapping[str, Any]) -> None:
        for item in equivalents:
            if item in config:
                self.__setattr__(equivalents[item], config[item])

    @override
    def __setattr__(self, item: str, value: Any) -> None:
        super().__setattr__(item, value)
        if item != "config":
            if isinstance(value, Settings):
                self.config[item] = value.config
            else:
                self.config[item] = value

    def __getattr__(self, item: str) -> Any:
        raise AttributeError(item)

    def __contains__(self, x: str) -> bool:
        return x in self.config

    def __getitem__(self, x: str) -> Any:
        return self.config.__getitem__(x)

    def __iter__(self) -> Iterator[str]:
        return self.config.__iter__()

    def __len__(self) -> int:
        return self.config.__len__()


def type_convert(inp: str) -> float | str | bool:
    inp = inp.strip()
    try:
        ret: float | str | bool = float(inp)
    except (ValueError, TypeError):
        if inp == "True":
            ret = True
        elif inp == "False":
            ret = False
        elif "$" in inp:
            ret = rf"{inp}"
        else:
            ret = inp
    return ret


base_path = os.sep.join(inspect.getfile(Settings).split(os.sep)[:-1])
settings = Settings(filename=base_path + os.sep + "default_settings.ini")
