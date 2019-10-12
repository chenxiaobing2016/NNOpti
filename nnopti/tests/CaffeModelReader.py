# -- coding:utf-8 --

import argparse
from utils.CaffeModelReader import parse_caffemodel


def test():
    parser = argparse.ArgumentParser(description="arguments of Neuron Network Optimizer")
    parser.add_argument('-n', '--network', default="data/ResNet-50-model.caffemodel")
    args = parser.parse_args()
    network = args.network
    parse_caffemodel(network)


if __name__ == "__main__":
    test()
