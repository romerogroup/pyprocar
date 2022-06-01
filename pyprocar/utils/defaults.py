# -*- coding: utf-8 -*-

import configparser
import inspect
import os
from abc import ABCMeta
from collections.abc import Mapping

equivalents = {'cmap': 'color_map',
               'cmaps': 'color_map',
               'ylim': 'elimit',
               'ylimit': 'elimit',
               'efermi': 'fermi',
               'mask': 'projection_mask',
               'marker_size': 'markersize',
               'colors':'color',
               'opacities':'opacity',
               'linewidths':'linewidth',
               'labels':'label'
               }



class Settings:
    __metaclass__ = ABCMeta

    def __init__(self, filename=None, config=None):
        self.config = {}
        if config is None and filename is not None:
            config = configparser.ConfigParser()
            config.read(filename)
            for item in config.sections():
                sub = Settings(config=config[item])
                self.__setattr__(item, sub)

        else:
            for item in config:
                if ',' in config[item]:
                    attr = config[item].split(',')
                    attr = [type_convert(x) for x in attr]
                else:
                    attr = type_convert(config[item])     
                self.__setattr__(item, attr)
        self.check_equivalents(config)

    def modify(self, changes):
        """Maybe needs modification to specify section to change"""
        changes = {item: changes[item] for item in changes }
        for item in changes:
            if item in self.config:
                self.__setattr__(item, changes[item])
            # else:
            #     for key in self.config:
            #         if item in self.config[key]:
            #             eval(
            #                 "self.{}.__setattr__(item, changes[item])".format(key))
            #             eval("self.{}.check_equivalents(changes)".format(key))

    def check_equivalents(self, config):
        for item in equivalents:
            if item in config:
                self.__setattr__(equivalents[item], config[item])

    def __setattr__(self, item, value):
        super().__setattr__(item, value)
        if item != 'config':
            if isinstance(value, Settings):
                self.config[item] = value.config
            else:
                self.config[item] = value

    def __contains__(self, x):
        return x in self.config

    def __getitem__(self, x):
        return self.config.__getitem__(x)

    def __iter__(self):
        return self.config.__iter__()

    def __len__(self):
        
        return self.config.__len__()

def type_convert(inp):
    inp = inp.strip()
    try:
        ret = float(inp)
    except BaseException:

        if inp == 'True':
            ret = True
        elif inp == 'False':
            ret = False
        elif "$" in inp:
            ret = r"{}".format(inp)
        else:
            ret = inp
    return ret


base_path = os.sep.join(inspect.getfile(Settings).split(os.sep)[:-1])
settings = Settings(filename= base_path + os.sep + 'default_settings.ini')
