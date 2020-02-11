import os
from importlib import import_module

from AbstractModel import AbstractModel


def save_model_architecture(res_dir, model):
    # save model config to json
    model_json = model.to_json()
    with open(os.path.join(res_dir, "model.json"), "w") as json_file:
        json_file.write(model_json)


def get_model_class(model_name: str) -> AbstractModel:
    return import_module("models.{}".format(model_name)).Model()


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle