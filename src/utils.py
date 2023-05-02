import json
import os.path

import pathlib


class Formatter:
    def format(self, the_object):
        return str(the_object)


class StringFormatter(Formatter):
    def format(self, the_object):
        return the_object


class JsonFormatter(Formatter):
    def format(self, the_object):
        return json.dumps(the_object)


FORMATTERS = {'.csv': StringFormatter(),
              '.txt': StringFormatter(),
              '.njson': JsonFormatter(),
              '.jsonl': JsonFormatter(),
              '.ndjson': JsonFormatter(),
              '.jsonlines': JsonFormatter(),
              '.jl': JsonFormatter(),
              }


def get_base_path():
    return pathlib.Path(__file__).parent.absolute()


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def to_file(store_path, stream, formatter=None):
    if not formatter:
        _, file_extension = os.path.splitext(store_path)
        formatter = FORMATTERS[file_extension]
    with open(store_path, 'w+') as f:
        for line in stream:
            f.write(formatter.format(line) + '\n')
