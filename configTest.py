# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging.config

from omegaconf import OmegaConf, DictConfig

import hydra
from hydra._internal.pathlib import Path
from hydra.plugins.common.utils import HydraConfig

log = logging.getLogger(__name__)

class Tester: 

    def __init__(self, cfg):
        self.cfg = cfg

        
    def get_method(self, path):
        return get_class(path)


    def get_class(self, path):
        try:
            from importlib import import_module

            module_path, _, class_name = path.rpartition(".")
            print(f"The path is : {path}")
            print(f"The module_path is : {module_path}")
            print(f"The class_name is: {class_name}")
            mod = import_module(module_path)
           
            try:
                klass = getattr(mod, class_name)
            except AttributeError:
                raise ImportError(
                    "Class {} is not in module {}".format(class_name, module_path)
                )
            return klass
        except ValueError as e:
            # module_path, _, class_name = path.rpartition(".")
            # mod = import_module(module_path)
            # print(f"The path is : {path}")
            # print(f"The module_path is : {module_path}")
            # print(f"The class_name is: {class_name}")
            log.error("Error initializing class " + path)
            raise e


    def get_static_method(self, full_method_name):
        try:
            spl = full_method_name.split(".")
            method_name = spl.pop()
            class_name = ".".join(spl)
            clz = self.get_class(class_name)
            return getattr(clz, method_name)
        except Exception as e:
            log.error("Error getting static method {} : {}".format(full_method_name, e))
            raise e


    def instantiate(self, config, *args, **kwargs):
        assert config is not None, "Input config is None"
        try:
            #print(config["class"])
            # return
            clazz = self.get_class(config["class"])
          
            params = config.params if "params" in config else OmegaConf.create()
            assert isinstance(
                params, DictConfig
            ), "Input config params are expected to be a mapping, found {}".format(
                type(config.params)
                )
            params.merge_with(OmegaConf.create(kwargs))
            return clazz(*args, **params)
        except Exception as e:
     
            log.error("Error instantiating {} : {}".format(config["class"], e))
            raise e


    def get_original_cwd(self):
        return HydraConfig().hydra.runtime.cwd


    def to_absolute_path(self, path):
        """
        converts the specified path to be absolute path.
        if the input path is relative, it's interpreted as relative to the original working directory
        if it's absolute, it's returned as is
        :param path:
        :return:
        """
        path = Path(path)
        if path.is_absolute():
            ret = path
        else:
            ret = Path(get_original_cwd()) / path
        return str(ret)

# @hydra.main(config_path='/home/yb1025/Research/GRAIL/relabeler-irl/accelerate-skillDiscovery/generalized-hindsight/rlkit/torch/diayn/config/train.yaml', strict=True)
@hydra.main(config_path='/home/yb1025/Research/GRAIL/relabeler-irl/accelerate-skillDiscovery/library-algo/diayn-main/config/train.yaml', strict=True)

def main(cfg):
    testerObj = Tester(cfg)
    # print("Tester object instantiated")
    testerObj.instantiate(cfg.agent)
                                
if __name__ == '__main__':
    main()
  