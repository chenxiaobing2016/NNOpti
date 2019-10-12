# -- coding:utf-8 --


class ProtoReader:

    def __init__(self):
        pass

    @staticmethod
    def read_proto(proto_file):
        with open(proto_file) as fin:
            lines = fin.readlines()
            return ProtoReader.parse(lines, 0, len(lines))

    @staticmethod
    def parse(lines, start, end):
        kv = dict()
        i = start
        depth = 0
        while i < end:
            line = lines[i].strip()
            assert len(line) == 0 or line[0] != '}'
            # skip blank line or comments
            if len(line) == 0 or line[0] == '#' or line[0] == '}':
                i += 1
            elif '{' in line:
                key = line.split('{')[0].strip()
                j = i + 1
                depth += 1
                while depth != 0:
                    if '{' in lines[j]:
                        depth += 1
                    elif '}' in lines[j]:
                        depth -= 1
                    j += 1
                # i + 1 point to the next line after '{', and
                # j - 1 point to '}'
                value = ProtoReader.parse(lines, i + 1, j - 1)
                # print('key: {0}  value: {1}'.format(key, value))
                if key in kv:
                    if type(kv[key]).__name__ == 'list':
                        kv[key].append(value)
                    else:
                        kv[key] = [kv[key], value]
                else:
                    kv[key] = value
                # update i to the next line after '}'
                i = j
            else:
                key, value = lines[i].split(':')
                key = key.strip()
                value = value.strip().replace('\"', '')
                if value == 'false':
                    value = False
                elif value == 'true':
                    value = True
                elif value.isdigit():
                    value = int(value)
                # print('key: {0}  value: {1}'.format(key, value))
                if key in kv:
                    if type(kv[key]).__name__ == 'list':
                        kv[key].append(value)
                    else:
                        kv[key] = [kv[key], value]
                else:
                    kv[key] = value
                i += 1
        return kv
