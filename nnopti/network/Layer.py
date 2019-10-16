# -- coding:utf-8 --

from enum import Enum
from network.Data import Data
import math


class LayerType(Enum):
    UNSET = 0
    CONVOLUTION = 1
    BATCH_NORM = 2
    SCALE = 3
    ELTWISE = 4
    POOL = 5
    INNER_PRODUCT = 6
    SOFTMAX = 7
    RELU = 8
    DROPOUT = 9

    @staticmethod
    def to_string(param):
        if param == LayerType.UNSET:
            return "UNSET"
        elif param == LayerType.CONVOLUTION:
            return "CONVOLUTION"
        elif param == LayerType.BATCH_NORM:
            return "BATCH_NORM"
        elif param == LayerType.SCALE:
            return "SCALE"
        elif param == LayerType.ELTWISE:
            return "ELTWISE"
        elif param == LayerType.POOL:
            return "POOL"
        elif param == LayerType.INNER_PRODUCT:
            return "INNER_PRODUCT"
        elif param == LayerType.SOFTMAX:
            return "SOFTMAX"
        elif param == LayerType.RELU:
            return "RELU"
        elif param == LayerType.DROPOUT:
            return "DROPOUT"
        else:
            raise Exception("Unknown layer type.")


class Layer(object):
    def __init__(self):
        self.layer_type = LayerType.UNSET
        self.name = ""

    def print_info(self, indent=0):
        print " " * indent + "layer name: %s" % self.name
        print " " * indent + "layer type: %s" % LayerType.to_string(self.layer_type)

    @staticmethod
    def create_from_json(json, data_set):
        assert "type" in json
        if json["type"] == "Convolution":
            layer = ConvLayer.create_from_json(json, data_set)
        elif json["type"] == "BatchNorm":
            layer = BatchNormLayer.create_from_json(json, data_set)
        elif json["type"] == "Scale":
            layer = ScaleLayer.create_from_json(json, data_set)
        elif json["type"] == "Eltwise":
            layer = EltwiseLayer.create_from_json(json, data_set)
        elif json["type"] == "Pooling":
            layer = PoolLayer.create_from_json(json, data_set)
        elif json["type"] == "InnerProduct":
            layer = InnerProductLayer.create_from_json(json, data_set)
        elif json["type"] == "Softmax":
            layer = SoftmaxLayer.create_from_json(json, data_set)
        elif json["type"] == "ReLU":
            layer = ReluLayer.create_from_json(json, data_set)
        elif json["type"] == "Dropout":
            layer = DropoutLayer.create_from_json(json, data_set)
        else:
            raise Exception("Unknown layer type")
        return layer

    def init_data_by_caffe_model(self, caffe_layer):
        pass


class ConvLayer(Layer):

    def __init__(self):
        Layer.__init__(self)
        self.pad = 0
        self.kx = 0
        self.ky = 0
        self.sx = 0
        self.sy = 0
        self.co = 0
        self.bottom = []
        self.top = []
        self.layer_type = LayerType.CONVOLUTION

    def print_info(self, indent=0):
        super(ConvLayer, self).print_info()
        print " " * indent + "convolution params:"
        print " " * indent + "  pad: %d" % self.pad
        print " " * indent + "  kx:  %d" % self.kx
        print " " * indent + "  ky:  %d" % self.ky
        print " " * indent + "  sx:  %d" % self.sx
        print " " * indent + "  sy:  %d" % self.sy
        print " " * indent + "bottom data:"
        self.bottom[0].print_info(2 + indent)
        print " " * indent + "top data:"
        self.top[0].print_info(2 + indent)
        print " " * indent + "weight data:"
        self.weight[0].print_info(2 + indent)
        if hasattr(self, 'bias'):
            print " " * indent + "bias data:"
            self.bias[0].print_info(2 + indent)


    @staticmethod
    def create_from_json(json, data_set):
        layer = ConvLayer()
        layer.layer_type = LayerType.CONVOLUTION
        layer.name = json["name"]

        if "convolution_param" in json:
            params = json["convolution_param"]
            if 'stride' in params:
                layer.sx = layer.sy = params['stride']
            if 'pad' in params:
                layer.pad = params['pad']
            if 'kernel_size' in params:
                layer.kx = layer.ky = params['kernel_size']
            if 'num_output' in params:
                layer.co = params['num_output']

        input_name = json["bottom"]
        input_data = Data()
        for data in data_set:
            if input_name == data.get_name():
                input_data = data
                break
        ni, ci, hi, wi = input_data.get_shape()
        no = ni
        ho = (hi + layer.pad * 2 - layer.ky) / layer.sy + 1
        wo = (wi + layer.pad * 2 - layer.kx) / layer.sx + 1
        output_name = json["top"]
        output_data = Data()
        output_data.set_name(output_name)
        output_data.set_shape([no, layer.co, ho, wo])
        data_set.append(output_data)
        weight_data = Data()
        weight_data.set_name(layer.name + "_weight")
        weight_data.set_shape([layer.co, ci, layer.kx, layer.ky])
        data_set.append(weight_data)
        layer.bottom = [input_data]
        layer.weight = [weight_data]
        layer.top = [output_data]
        if not (('convolution_param') in json and ('bias_term' in json['convolution_param']) and (json['convolution_param']['bias_term'] is False)):
            bias_data = Data()
            bias_data.set_name(layer.name + "_bias")
            bias_data.set_shape([1, layer.co, 1, 1])
            data_set.append(bias_data)
            layer.bias = [bias_data]
        return layer

    def init_data_by_caffe_model(self, caffe_layer):
        self.weight[0].set_content(caffe_layer.blobs[0].data)
        if hasattr(self, 'bias'):
            self.bias[0].set_content(caffe_layer.blobs[1].data)


class BatchNormLayer(Layer):

    def __init__(self):
        Layer.__init__(self)
        self.bottom = []
        self.top = []
        self.mean = []
        self.vari = []
        self.layer_type = LayerType.BATCH_NORM

    def print_info(self, indent=0):
        super(BatchNormLayer, self).print_info()
        print " " * indent + "bottom data:"
        self.bottom[0].print_info(2 + indent)
        print " " * indent + "top data:"
        self.top[0].print_info(2 + indent)
        print " " * indent + "mean data:"
        self.mean[0].print_info(2 + indent)
        print " " * indent + "variance data:"
        self.vari[0].print_info(2 + indent)

    @staticmethod
    def create_from_json(json, data_set):
        layer = BatchNormLayer()
        layer.name = json["name"]

        input_name = json["bottom"]
        input_data = Data()
        for data in data_set:
            if input_name == data.get_name():
                input_data = data
                break
        ni, ci, hi, wi = input_data.get_shape()
        no = ni
        co = ci
        ho = hi
        wo = wi
        output_name = json["top"]
        output_data = Data()
        output_data.set_name(output_name)
        output_data.set_shape([no, co, ho, wo])
        data_set.append(output_data)
        mean_data = Data()
        mean_data.set_name(layer.name + "_mean")
        mean_data.set_shape([1, co, 1, 1])
        data_set.append(mean_data)
        vari_data = Data()
        vari_data.set_name(layer.name + "_vari")
        vari_data.set_shape([1, co, 1, 1])
        data_set.append(vari_data)

        layer.bottom = [input_data]
        layer.vari = [vari_data]
        layer.mean = [mean_data]
        layer.top = [output_data]
        return layer

    def init_data_by_caffe_model(self, caffe_layer):
        self.mean[0].set_content(caffe_layer.blobs[0].data)
        self.vari[0].set_content(caffe_layer.blobs[1].data)


class ScaleLayer(Layer):

    def __init__(self):
        Layer.__init__(self)
        self.bottom = []
        self.top = []
        self.alpha = []
        self.beta = []
        self.layer_type = LayerType.SCALE

    def print_info(self, indent=0):
        super(ScaleLayer, self).print_info()
        print " " * indent + "bottom data:"
        self.bottom[0].print_info(2 + indent)
        print " " * indent + "top data:"
        self.top[0].print_info(2 + indent)
        print " " * indent + "alpha data:"
        self.alpha[0].print_info(2 + indent)
        print " " * indent + "beta data:"
        self.beta[0].print_info(2 + indent)

    @staticmethod
    def create_from_json(json, data_set):
        layer = ScaleLayer()
        layer.name = json["name"]

        input_name = json["bottom"]
        input_data = Data()
        for data in data_set:
            if input_name == data.get_name():
                input_data = data
                break
        ni, ci, hi, wi = input_data.get_shape()
        no = ni
        co = ci
        ho = hi
        wo = wi
        output_name = json["top"]
        output_data = Data()
        output_data.set_name(output_name)
        output_data.set_shape([no, co, ho, wo])
        data_set.append(output_data)
        alpha_data = Data()
        alpha_data.set_name(layer.name + "_alpha")
        alpha_data.set_shape([1, co, 1, 1])
        data_set.append(alpha_data)
        beta_data = Data()
        beta_data.set_name(layer.name + "_beta")
        beta_data.set_shape([1, co, 1, 1])
        data_set.append(beta_data)

        layer.bottom = [input_data]
        layer.alpha = [alpha_data]
        layer.beta = [beta_data]
        layer.top = [output_data]
        return layer

    def init_data_by_caffe_model(self, caffe_layer):
        self.alpha[0].set_content(caffe_layer.blobs[0].data)
        self.beta[0].set_content(caffe_layer.blobs[1].data)


class EltwiseLayer(Layer):

    def __init__(self):
        Layer.__init__(self)
        self.bottom = []
        self.top = []
        self.layer_type = LayerType.ELTWISE

    def print_info(self, indent=0):
        super(EltwiseLayer, self).print_info()
        for i in range(len(self.bottom)):
            print " " * indent + "bottom data %d:" % i
            self.bottom[0].print_info(2 + indent)
        print " " * indent + "top data:"
        self.top[0].print_info(2 + indent)

    @staticmethod
    def create_from_json(json, data_set):
        layer = EltwiseLayer()
        layer.name = json["name"]

        input_names = json["bottom"]
        input_data = []
        for input_name in input_names:
            for data in data_set:
                if input_name == data.get_name():
                    input_data.append(data)
                    break

        ni, ci, hi, wi = input_data[0].get_shape()
        no = ni
        co = ci
        ho = hi
        wo = wi
        output_name = json["top"]
        output_data = Data()
        output_data.set_name(output_name)
        output_data.set_shape([no, co, ho, wo])
        data_set.append(output_data)

        layer.bottom = input_data
        layer.top = [output_data]
        return layer


class PoolLayer(Layer):
    class PoolType(Enum):
        UNSET = 0
        AVG = 1
        MAX = 2

        @staticmethod
        def to_string(pool_type):
            if pool_type == PoolLayer.PoolType.UNSET:
                return "UNSET"
            elif pool_type == PoolLayer.PoolType.AVG:
                return "AVG"
            elif pool_type == PoolLayer.PoolType.MAX:
                return "MAX"
            else:
                raise Exception("Unknown pool type")

    def __init__(self):
        Layer.__init__(self)
        self.bottom = []
        self.top = []
        self.layer_type = LayerType.POOL
        self.pad = 0
        self.sx = 0
        self.sy = 0
        self.kx = 0
        self.ky = 0
        self.pool_type = PoolLayer.PoolType.UNSET

    def print_info(self, indent=0):
        super(PoolLayer, self).print_info()
        print " " * indent + "pool params:"
        print " " * indent + "  pool type: %s" % PoolLayer.PoolType.to_string(self.pool_type)
        print " " * indent + "  pad: %d" % self.pad
        print " " * indent + "  kx:  %d" % self.kx
        print " " * indent + "  ky:  %d" % self.ky
        print " " * indent + "  sx:  %d" % self.sx
        print " " * indent + "  sy:  %d" % self.sy
        print " " * indent + "bottom data:"
        self.bottom[0].print_info(2 + indent)
        print " " * indent + "top data:"
        self.top[0].print_info(2 + indent)

    @staticmethod
    def create_from_json(json, data_set):
        layer = PoolLayer()
        layer.name = json["name"]
        params = json["pooling_param"]
        layer.sx = layer.sy = params["stride"]
        layer.kx = layer.ky = params["kernel_size"]
        if params['pool'] == 'MAX':
            layer.pool_type = PoolLayer.PoolType.MAX
        elif params['pool'] == 'AVE':
            layer.pool_type = PoolLayer.PoolType.AVG
        else:
            raise Exception("Unknown pool type")

        input_name = json["bottom"]
        input_data = Data()
        for data in data_set:
            if input_name == data.get_name():
                input_data = data
                break
        ni, ci, hi, wi = input_data.get_shape()
        no = ni
        co = ci
        ho = int(math.ceil((hi - layer.ky) * 1. / layer.sy + 1))
        wo = int(math.ceil((wi - layer.kx) * 1. / layer.sx + 1))
        output_name = json["top"]
        output_data = Data()
        output_data.set_name(output_name)
        output_data.set_shape([no, co, ho, wo])
        data_set.append(output_data)
        layer.bottom = [input_data]
        layer.top = [output_data]

        return layer


class InnerProductLayer(Layer):

    def __init__(self):
        Layer.__init__(self)
        self.kx = 1
        self.ky = 1
        self.co = 0
        self.bottom = []
        self.top = []
        self.layer_type = LayerType.INNER_PRODUCT

    def print_info(self, indent=0):
        super(InnerProductLayer, self).print_info()
        print " " * indent + "inner product params:"
        print " " * indent + "  kx:  %d" % self.kx
        print " " * indent + "  ky:  %d" % self.ky
        print " " * indent + "bottom data:"
        self.bottom[0].print_info(2 + indent)
        print " " * indent + "top data:"
        self.top[0].print_info(2 + indent)
        print " " * indent + "weight data:"
        self.weight[0].print_info(2 + indent)
        if hasattr(self, 'bias'):
            print " " * indent + "bias data:"
            self.bias[0].print_info(2 + indent)


    @staticmethod
    def create_from_json(json, data_set):
        layer = InnerProductLayer()
        layer.name = json["name"]
        layer.layer_type = LayerType.INNER_PRODUCT

        if "inner_product_param" in json:
            params = json["inner_product_param"]
            if 'num_output' in params:
                layer.co = params['num_output']

        input_name = json["bottom"]
        input_data = Data()
        for data in data_set:
            if input_name == data.get_name():
                input_data = data
                break
        ni, ci, hi, wi = input_data.get_shape()
        no = ni
        output_name = json["top"]
        output_data = Data()
        output_data.set_name(output_name)
        output_data.set_shape([ni, layer.co, 1, 1])
        data_set.append(output_data)
        weight_data = Data()
        weight_data.set_name(layer.name + "_weight")
        weight_data.set_shape([layer.co, ci, hi, wi])
        data_set.append(weight_data)
        layer.bottom = [input_data]
        layer.weight = [weight_data]
        layer.top = [output_data]
        if not (('inner_product_param') in json and ('bias_term' in json['inner_product_param']) and (json['inner_product_param']['bias_term'] is False)):
            bias_data = Data()
            bias_data.set_name(layer.name + "_bias")
            bias_data.set_shape([1, layer.co, 1, 1])
            layer.bias = [bias_data]
            data_set.append(bias_data)
        return layer

    def init_data_by_caffe_model(self, caffe_layer):
        self.weight[0].set_content(caffe_layer.blobs[0].data)
        if hasattr(self, 'bias'):
            self.bias[0].set_content(caffe_layer.blobs[1].data)


class SoftmaxLayer(Layer):

    def __init__(self):
        Layer.__init__(self)
        self.bottom = []
        self.top = []
        self.layer_type = LayerType.SOFTMAX

    def print_info(self, indent=0):
        super(SoftmaxLayer, self).print_info()
        print " " * indent + "bottom data:"
        self.bottom[0].print_info(2 + indent)
        print " " * indent + "top data:"
        self.top[0].print_info(2 + indent)

    @staticmethod
    def create_from_json(json, data_set):
        layer = SoftmaxLayer()
        layer.name = json["name"]

        input_name = json["bottom"]
        input_data = Data()
        for data in data_set:
            if input_name == data.get_name():
                input_data = data
                break
        ni, ci, hi, wi = input_data.get_shape()
        no = ni
        co = ci
        ho = hi
        wo = wi
        output_name = json["top"]
        output_data = Data()
        output_data.set_name(output_name)
        output_data.set_shape([no, co, ho, wo])
        data_set.append(output_data)

        layer.bottom = [input_data]
        layer.top = [output_data]
        return layer


class ReluLayer(Layer):

    def __init__(self):
        Layer.__init__(self)
        self.bottom = []
        self.top = []
        self.layer_type = LayerType.RELU

    def print_info(self, indent=0):
        super(ReluLayer, self).print_info()
        print " " * indent + "bottom data:"
        self.bottom[0].print_info(2 + indent)
        print " " * indent + "top data:"
        self.top[0].print_info(2 + indent)

    @staticmethod
    def create_from_json(json, data_set):
        layer = ReluLayer()
        layer.name = json["name"]

        input_name = json["bottom"]
        input_data = Data()
        for data in data_set:
            if input_name == data.get_name():
                input_data = data
                break
        ni, ci, hi, wi = input_data.get_shape()
        no = ni
        co = ci
        ho = hi
        wo = wi
        output_name = json["top"]
        output_data = Data()
        output_data.set_name(output_name)
        output_data.set_shape([no, co, ho, wo])
        data_set.append(output_data)

        layer.bottom = [input_data]
        layer.top = [output_data]
        return layer


class DropoutLayer(Layer):

    def __init__(self):
        Layer.__init__(self)
        self.bottom = []
        self.top = []
        self.layer_type = LayerType.DROPOUT

    def print_info(self, indent=0):
        super(DropoutLayer, self).print_info()
        print " " * indent + "bottom data:"
        self.bottom[0].print_info(2 + indent)
        print " " * indent + "top data:"
        self.top[0].print_info(2 + indent)

    @staticmethod
    def create_from_json(json, data_set):
        layer = ReluLayer()
        layer.name = json["name"]

        input_name = json["bottom"]
        input_data = Data()
        for data in data_set:
            if input_name == data.get_name():
                input_data = data
                break
        ni, ci, hi, wi = input_data.get_shape()
        no = ni
        co = ci
        ho = hi
        wo = wi
        output_name = json["top"]
        output_data = Data()
        output_data.set_name(output_name)
        output_data.set_shape([no, co, ho, wo])
        data_set.append(output_data)
        if "dropout_param" in json:
            if "dropout_ratio" in json["dropout_param"]:
                self.dropout_ratio == json["dropout_param"]["dropout_ratio"]

        layer.bottom = [input_data]
        layer.top = [output_data]
        return layer
