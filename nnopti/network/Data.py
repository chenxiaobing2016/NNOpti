# -- coding:utf-8 --

import numpy as np


class Data:

    def __init__(self):
        self.shape = []
        self.name = ""
        self.content = np.empty(0)

    def set_shape(self, shape):
        self.shape = shape

    def get_shape(self):
        return self.shape

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_content(self, content):
        self.content = np.array(content)
        self.content.resize(self.shape)

    def get_content(self):
        return self.content

    def print_info(self, indent=0):
        print " " * indent + "name: %s" % self.name
        print " " * indent + "shape: %s" % self.shape.__str__()
