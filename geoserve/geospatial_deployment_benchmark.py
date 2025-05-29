import argparse
import asyncio
import copy
import datetime
import importlib
import re
import uuid
from io import BytesIO
from typing import List, Union, Tuple, Any, Coroutine
import time

import aiohttp
import albumentations
import rasterio
import requests
import torch
from einops import rearrange
from ray import serve
from ray.serve.handle import DeploymentHandle
from terratorch.datamodules import Sen1Floods11NonGeoDataModule
from torch import Tensor
from vllm import SamplingParams, PoolingRequestOutput
import numpy as np


NO_DATA = -9999
NO_DATA_FLOAT = 0.0001

@serve.deployment
class GeospatialDeploymentBenchamrk():

    def __init__(
            self, vllm_deployment: DeploymentHandle,
            args: argparse.Namespace,
    ):
        self.vllmDeployment = vllm_deployment.options(stream=True)
        self.id = str(uuid.uuid4())
        self.datamodule = self.generate_datamodule_static()
        self.args = args
        self.extra_data = self.alocate_random_bytes(args.extra_data_size) if args.extra_data_size > 0 else None

    def get_sleep_time(self):
        if self.args.sleep_distribution == "uniform":
            return np.random.uniform(self.args.sleep_min, self.args.sleep_max)
        elif self.args.sleep_distribution == "fixed":
            return self.args.sleep_fixed
        else:
            return 0.0

    def alocate_random_bytes(self, size: int):
        """Allocate random bytes of size *size*."""
        return np.random.default_rng().bytes(size)

    async def encode(self, geotiff_path: str) -> tuple[Tensor, dict[str, int]]:
        # This may be doing some unnecessary data movement
        # preprocessor_response = await self.preprocessorDeployment.apply.remote(geotiff_path)
        # Use it locally, avoid unnecessary data movement
        start_encode = time.time_ns()
        preprocessor_response = await self.apply(geotiff_path)
        end_preprocessor = time.time_ns()

        sleep_time = self.get_sleep_time()
        if sleep_time > 0.0:
            await asyncio.sleep(sleep_time)
        end_sleep = time.time_ns()

        location_coords = preprocessor_response["location_coords"]
        pixel_values_chunks = preprocessor_response["pixel_values_chunks"]

        img_size = preprocessor_response["img_size"]
        h1 = preprocessor_response["h1"]
        w1 = preprocessor_response["w1"]
        original_h = preprocessor_response["original_h"]
        original_w = preprocessor_response["original_w"]

        submission_timestamps = []
        pred_futures = []
        for pixel_values in pixel_values_chunks:
            mm_data = {
                "pixel_values": torch.empty(0) if pixel_values is None else pixel_values,
                "location_coords": torch.empty(0) if location_coords is None else location_coords,
                "temporal_coords": torch.empty(0),
            }

            if self.extra_data is not None:
                mm_data["extra_data"] = self.extra_data

            prompt = {
                "prompt_token_ids": [1],
                "multi_modal_data": mm_data
            }

            pred = self.vllmDeployment.encode.remote(prompt, SamplingParams(temperature=0.0), uuid.uuid4())
            pred_futures.append(pred)
            submission_timestamps.append(time.time_ns())

        submitted_all_vllm_tasks = time.time_ns()

        reception_timestamps = []
        pred_imgs = []
        for f in pred_futures:
            request_output: PoolingRequestOutput = await f.__anext__()
            reception_timestamps.append(time.time_ns())
            pred = request_output.outputs.data

            y_hat = pred.argmax(dim=1)
            y_hat = torch.nn.functional.interpolate(y_hat.unsqueeze(1).float(), size=img_size, mode="nearest")
            pred_imgs.append(y_hat)

        end_vllm = time.time_ns()

        pred_imgs = torch.concat(pred_imgs, dim=0)

        # Build images from patches
        pred_imgs = rearrange(
            pred_imgs,
            "(b h1 w1) c h w -> b c (h1 h) (w1 w)",
            h=img_size,
            w=img_size,
            b=1,
            c=1,
            h1=h1,
            w1=w1,
        )

        # Cut padded area back to original size
        pred_imgs = pred_imgs[..., :original_h, :original_w]

        # Squeeze (batch size 1)
        pred_imgs = pred_imgs[0]

        end_encode = time.time_ns()

        timestamps = {
            "start_encode": start_encode,
            "end_preprocessor": end_preprocessor,
            "end_sleep": end_sleep,
            "submission_timestamps": submission_timestamps,
            "submitted_all_vllm_tasks": submitted_all_vllm_tasks,
            "reception_timestamps": reception_timestamps,
            "end_vllm": end_vllm,
            "end_encode": end_encode,
        }

        return pred_imgs, timestamps


    def init_object_from_classpath_and_args(self, class_name: str, init_args: dict):
        class_parts = class_name.split('.')
        module_name = '.'.join(class_parts[:-1])
        class_name = class_parts[-1]

        # Import the module
        module = importlib.import_module(module_name)

        # Get the class
        cls = getattr(module, class_name)

        return cls(**init_args)

    def parse_args_list(self, args: list):
        return [
            self.init_object_from_classpath_and_args(t["class_path"], t["init_args"])
            for t in args]

    def generate_datamodule(self, classpath: str, init_args: dict) -> Sen1Floods11NonGeoDataModule:
        final_init_args = copy.deepcopy(init_args)

        final_init_args["test_transform"] = self.parse_args_list(init_args["test_transform"])
        final_init_args["val_transform"] = self.parse_args_list(init_args["val_transform"])
        final_init_args["train_transform"] = self.parse_args_list(init_args["test_transform"])

        datamodule = self.init_object_from_classpath_and_args(classpath, final_init_args)
        return datamodule

    def _preprocess(self, x: np.ndarray, img_size: int, location_coords: np.ndarray):
        # Reflect pad if not divisible by img_size
        original_h, original_w = x.shape[-2:]
        pad_h = (img_size - (original_h % img_size)) % img_size
        pad_w = (img_size - (original_w % img_size)) % img_size
        x = np.pad(
            x, ((0, 0), (0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="reflect"
        )

        # Build sliding window

        batch_size = 1
        batch = torch.tensor(x, device="cpu")
        windows = batch.unfold(3, img_size, img_size).unfold(4, img_size, img_size)
        h1, w1 = windows.shape[3:5]
        windows = rearrange(
            windows, "b c t h1 w1 h w -> (b h1 w1) c t h w", h=img_size, w=img_size
        )

        # Split into batches if number of windows > batch_size
        num_batches = windows.shape[0] // batch_size if windows.shape[0] > batch_size else 1
        windows = torch.tensor_split(windows, num_batches, dim=0)

        chunks = []
        for x in windows:
            # Apply standardization
            x = self.datamodule.test_transform(image=x.squeeze().numpy().transpose(1, 2, 0))
            x = self.datamodule.aug(x)['image']

            chunks.append(x)

        return {
            "pixel_values_chunks": chunks,
            "location_coords": torch.empty(0) if location_coords is None else location_coords,
            "img_size": img_size,
            "h1": h1,
            "w1": w1,
            "original_h": original_h,
            "original_w": original_w,
        }

    @staticmethod
    def generate_datamodule_static() -> Sen1Floods11NonGeoDataModule:
        datamodule_config = {
            'bands': ['BLUE',
                      'GREEN',
                      'RED',
                      'NIR_NARROW',
                      'SWIR_1',
                      'SWIR_2'],
            'batch_size': 16,
            'constant_scale': 0.0001,
            'data_root': '/dccstor/geofm-finetuning/datasets/sen1floods11',
            'drop_last': True,
            'no_data_replace': 0.0,
            'no_label_replace': -1,
            'num_workers': 8,
            'test_transform': [albumentations.Resize(always_apply=False,
                                                     height=448,
                                                     interpolation=1,
                                                     p=1,
                                                     width=448),
                               albumentations.pytorch.ToTensorV2(
                                   transpose_mask=False,
                                   always_apply=True,
                                   p=1.0
                               )],
        }

        datamodule = Sen1Floods11NonGeoDataModule(data_root=datamodule_config['data_root'],
                                                  batch_size=datamodule_config["batch_size"],
                                                  num_workers=datamodule_config["num_workers"],
                                                  bands=datamodule_config["bands"],
                                                  drop_last=datamodule_config["drop_last"],
                                                  test_transform=datamodule_config["test_transform"
                                                                                   ""])

        return datamodule

    @staticmethod
    def read_geotiff(file_path: str) -> Tuple[np.ndarray, dict, Tuple[float, float]]:
        """Read all bands from *file_path* and return image + meta info.

        Args:
            file_path: path to image file.

        Returns:
            np.ndarray with shape (bands, height, width)
            meta info dict
        """

        if file_path.startswith("http"):
            response = requests.get(file_path)
            response.raise_for_status()  # Raise an error for bad responses
            file = BytesIO(response.content)
        else:
            file = file_path

        with rasterio.open(file) as src:
            img = src.read()
            meta = src.meta
            try:
                coords = src.lnglat()
            except:
                # Cannot read coords
                coords = None
        return img, meta, coords

    @staticmethod
    async def read_geotiff_async(file_path: str) -> Tuple[np.ndarray, dict, Tuple[float, float]]:
        """Read all bands from *file_path* and return image + meta info.

        Args:
            file_path: path to image file.

        Returns:
            np.ndarray with shape (bands, height, width)
            meta info dict
        """

        if file_path.startswith("http"):
            async with aiohttp.ClientSession() as session:
                async with session.get(file_path) as response:
                    response.raise_for_status()  # Raise an error for bad responses
                    file = BytesIO(await response.read())
        else:
            file = file_path

        with rasterio.open(file) as src:
            img = src.read()
            meta = src.meta
            try:
                coords = src.lnglat()
            except:
                # Cannot read coords
                coords = None
        return img, meta, coords

    async def load_example(
            self,
            file_paths: List[str],
            mean: List[float] = None,
            std: List[float] = None,
            indices: Union[list[int], None] = None,
    ):
        """Build an input example by loading images in *file_paths*.

        Args:
            file_paths: list of file paths .
            mean: list containing mean values for each band in the images in *file_paths*.
            std: list containing std values for each band in the images in *file_paths*.

        Returns:
            np.array containing created example
            list of meta info for each image in *file_paths*
        """

        imgs = []
        metas = []
        temporal_coords = []
        location_coords = []

        for file in file_paths:
            img, meta, coords = await self.read_geotiff_async(file)

            # Rescaling (don't normalize on nodata)
            img = np.moveaxis(img, 0, -1)  # channels last for rescaling
            if indices is not None:
                img = img[..., indices]
            if mean is not None and std is not None:
                img = np.where(img == NO_DATA, NO_DATA_FLOAT, (img - mean) / std)

            imgs.append(img)
            metas.append(meta)
            if coords is not None:
                location_coords.append(coords)

            try:
                match = re.search(r'(\d{7,8}T\d{6})', file)
                if match:
                    year = int(match.group(1)[:4])
                    julian_day = match.group(1).split('T')[0][4:]
                    if len(julian_day) == 3:
                        julian_day = int(julian_day)
                    else:
                        julian_day = datetime.datetime.strptime(julian_day, '%m%d').timetuple().tm_yday
                    temporal_coords.append([year, julian_day])
            except Exception as e:
                print(f'Could not extract timestamp for {file} ({e})')

        imgs = np.stack(imgs, axis=0)  # num_frames, H, W, C
        imgs = np.moveaxis(imgs, -1, 0).astype("float32")  # C, num_frames, H, W
        imgs = np.expand_dims(imgs, axis=0)

        return imgs, temporal_coords, location_coords, metas

    async def apply(self, geotiff_path: str = None, input_data: np.ndarray = None,
              location_coords: torch.tensor = torch.empty(0), img_size: int = 512):
        if input_data is None and geotiff_path is None:
            raise ValueError("Either input_data or geotiff_path must be provided")


        if input_data is None:
            input_data, _, location_coords, _ = await self.load_example(file_paths=[geotiff_path], indices=[1,2,3,8,11,12])

        if input_data.mean() > 1:
            input_data = input_data / 10000

        return self._preprocess(input_data, img_size, location_coords)