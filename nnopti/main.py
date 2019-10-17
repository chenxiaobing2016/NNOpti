# -- coding:utf-8 --

import argparse
from utils.ProtoReader import ProtoReader
from utils.CaffeModelReader import parse_caffemodel
from network.Network import Network

from tests.ProtoReader import test as  proto_reader_test
from tests.CaffeModelReader import test as caffe_model_reader_test
import numpy as np


def test():
    proto_reader_test()
    caffe_model_reader_test()


def main():
    parser = argparse.ArgumentParser(description="arguments of Neuron Network Optimizer")
    parser.add_argument('-n', '--network', default="data/ResNet-50-deploy.prototxt")
    parser.add_argument('-m', '--model', default='data/ResNet-50-model.caffemodel')
    args = parser.parse_args()
    network_file = args.network
    json_file = ProtoReader.read_proto(network_file)

    model_file = args.model
    model = parse_caffemodel(model_file)

    network = Network()
    network.construct_from_json(json_file)
    network.init_caffe_model(model)

    in_data = network.input
    in_data.set_content(np.zeros(in_data.get_shape()))
    for layer in network.layers:
        print 'start execute layer %s' % layer.name
        layer.execute()
        layer.print_info()
        print ""
    idx = len(network.layers) - 1
    print "output:"
    print network.layers[idx].top[0].get_content()


if __name__ == "__main__":
    main()
