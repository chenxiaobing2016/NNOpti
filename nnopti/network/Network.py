# -- coding:utf-8 --

from network.Data import Data
from network.Layer import Layer


class Network:

    def __init__(self):
        self.name = ""
        self.input = Data()
        self.layers = []

    def construct_from_json(self, json):
        if "name" in json:
            self.name = json["name"]

        assert "input" in json
        self.input = Data()
        self.input.name = json['input']
        self.input.set_shape(json["input_dim"])

        data_set = [self.input]
        for jlayer in json["layer"]:
            # infer shape of output is conducted in the procedure
            layer = Layer.create_from_json(jlayer, data_set)
            self.layers.append(layer)

    def init_caffe_model(self, model):
        for layer_id in range(len(self.layers)):
            name = self.layers[layer_id].name
            # print 'layer id: %d layer name: %s' % (layer_id, name)
            find = False
            for i in range(len(model.layer)):
                if name == model.layer[i].name:
                    self.layers[layer_id].init_data_by_caffe_model(model.layer[i])
                    find = True
                    break
            if find is False:
                print 'layer %s is not found' % name

