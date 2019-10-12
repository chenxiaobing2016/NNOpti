# -- coding:utf-8 --

from tests.ProtoReader import test as  proto_reader_test
from tests.CaffeModelReader import test as caffe_model_reader_test


def main():
    proto_reader_test()
    caffe_model_reader_test()


if __name__ == "__main__":
    main()
