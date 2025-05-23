import time

import numpy as np
import torch
from ray import serve

from geoserve.geospatial_preprocessing import PrithviPreprocessor


@serve.deployment
class PrithviBenchmarkPreprocessor(PrithviPreprocessor):
    def __init__(self, config=None, extra_data_size=0, sleep_time=0):
        super().__init__(config)
        self.extra_data_size = extra_data_size
        self.sleep_time = sleep_time

    def apply(self, geotiff_path: str = None, input_data: np.ndarray = None,
              location_coords: torch.tensor = torch.empty(0), img_size: int = 512):

        res = super().apply(geotiff_path, input_data, location_coords, img_size)

        if self.sleep_time:
            time.sleep(self.sleep_time)

        if self.extra_data_size:
            # Simulate extra data processing
            extra_bytes = np.random.default_rng().bytes(self.extra_data_size)
            res["extra_data"] = extra_bytes

        return res
