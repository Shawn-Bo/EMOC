import logging
import os

import torch.cuda


class DeviceHandler(object):
    def __init__(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_device(self, cuda_visible_devices="0", cpu=False, display_log=False):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
        if display_log:
            self.logger.info(f"Device: {device}")
        return device

    @staticmethod
    def get_cuda_n_gpu():
        return torch.cuda.device_count()


deviceHandler = DeviceHandler()

if __name__ == "__main__":
    from CLIParser import argParser

    print(deviceHandler.get_device(argParser.parse_args()))
