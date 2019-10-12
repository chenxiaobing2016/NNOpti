# -- coding:utf-8 --

import argparse
from utils.ProtoReader import ProtoReader


def test():
    parser = argparse.ArgumentParser(description="arguments of Neuron Network Optimizer")
    parser.add_argument('-n', '--network', default="data/ResNet-50-deploy.prototxt")
    args = parser.parse_args()
    network = args.network
    print ProtoReader.read_proto(network)


if __name__ == "__main__":
    test()
