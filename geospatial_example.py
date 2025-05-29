import datetime
import os
import re
from io import BytesIO
from typing import Union, List, Tuple

import numpy as np
import rasterio
import ray
import requests
import aiohttp
import torch
import yaml
from ray import serve
from ray.serve.handle import DeploymentHandle
from vllm import AsyncEngineArgs

from geoserve.async_llm_engine_support import AsyncLLMEngineDeployment
from geoserve.geospatial_deployment import GeospatialDeployment
from geoserve.geospatial_preprocessing import PrithviPreprocessorDeployment

NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
OFFSET = 0
PERCENTILE = 99


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

def load_example(
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
        img, meta, coords = read_geotiff(file)

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

def process_channel_group(orig_img, channels):
    """
    Args:
        orig_img: torch.Tensor representing original image (reference) with shape = (bands, H, W).
        channels: list of indices representing RGB channels.

    Returns:
        torch.Tensor with shape (num_channels, height, width) for original image
    """

    orig_img = orig_img[channels, ...]
    valid_mask = torch.ones_like(orig_img, dtype=torch.bool)
    valid_mask[orig_img == NO_DATA_FLOAT] = False


    # Rescale (enhancing contrast)
    max_value = max(3000, np.percentile(orig_img[valid_mask], PERCENTILE))
    min_value = OFFSET

    orig_img = torch.clamp((orig_img - min_value) / (max_value - min_value), 0, 1)

    # No data as zeros
    orig_img[~valid_mask] = 0

    return orig_img


def save_geotiff(image, output_path: str, meta: dict):
    """Save multi-band image in Geotiff file.

    Args:
        image: np.ndarray with shape (bands, height, width)
        output_path: path where to save the image
        meta: dict with meta info.
    """

    with rasterio.open(output_path, "w", **meta) as dest:
        for i in range(image.shape[0]):
            dest.write(image[i, :, :], i + 1)

    return


def _convert_np_uint8(float_image: torch.Tensor):
    image = float_image.numpy() * 255.0
    image = image.astype(dtype=np.uint8)

    return image


def main(
            data_file: str,
            config: str,
            output_dir: str,
            rgb_outputs: bool = False,
            input_indices: list[int] = None,
    ):
    args = AsyncEngineArgs(model="christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM", skip_tokenizer_init=True, dtype="float32")
    vllm_engine = AsyncLLMEngineDeployment.bind(engine_args=args)
    # preprocessing = PrithviPreprocessorDeployment.bind()
    model = GeospatialDeployment.bind(vllm_deployment=vllm_engine)

    try:
        handle: DeploymentHandle = serve.run(model, name="geospatial-prithvi-example", route_prefix=None)


        os.makedirs(output_dir, exist_ok=True)

        with open(config, "r") as f:
            config_dict = yaml.safe_load(f)

        # Loading data ---------------------------------------------------------------------------------
        input_data, temporal_coords, location_coords, meta_data = load_example(
            file_paths=[data_file], indices=input_indices,
        )

        meta_data = meta_data[0]  # only one image

        if input_data.mean() > 1:
            input_data = input_data / 10000  # Convert to range 0-1

        # Running model --------------------------------------------------------------------------------

        channels = [config_dict['data']['init_args']['bands'].index(b) for b in ["RED", "GREEN", "BLUE"]]  # BGR -> RGB

        pred = handle.encode.remote(data_file).result()

        # print(pred)

        # Save pred
        meta_data.update(count=1, dtype="uint8", compress="lzw", nodata=0)
        pred_file = os.path.join(output_dir, f"pred_{os.path.splitext(os.path.basename(data_file))[0]}.tiff")
        save_geotiff(_convert_np_uint8(pred), pred_file, meta_data)

        # Save image + pred
        meta_data.update(count=3, dtype="uint8", compress="lzw", nodata=0)

        if input_data.mean() < 1:
            input_data = input_data * 10000  # Scale to 0-10000

        rgb_orig = process_channel_group(
            orig_img=torch.Tensor(input_data[0, :, 0, ...]),
            channels=channels,
        )

        pred[pred == 0.] = np.nan
        img_pred = rgb_orig * 0.7 + pred * 0.3
        img_pred[img_pred.isnan()] = rgb_orig[img_pred.isnan()]

        img_pred_file = os.path.join(output_dir, f"rgb_pred_{os.path.splitext(os.path.basename(data_file))[0]}.tiff")
        save_geotiff(
            image=_convert_np_uint8(img_pred),
            output_path=img_pred_file,
            meta=meta_data,
        )

        # Save image rgb
        if rgb_outputs:
            rgb_file = os.path.join(output_dir, f"original_rgb_{os.path.splitext(os.path.basename(data_file))[0]}.tiff")
            save_geotiff(
                image=_convert_np_uint8(rgb_orig),
                output_path=rgb_file,
                meta=meta_data,
            )

        print("Done!")

    except Exception as e:
        print(f"Error: {e.with_traceback()}")

    finally:
        # Clean up
        serve.delete("geospatial-prithvi-example")
        print("Cleaned up resources.")


if __name__ == "__main__":
    ray_address = "ray://localhost:10001"
    geotiff_file = "data/India_900498_S2Hand.tif"
    config = "./model/config.yaml"
    output_dir = "./output"
    runtime_env = {"pip": "requirements.txt"}
    ray.init(address=ray_address, runtime_env=runtime_env)
    main(geotiff_file, config, output_dir)