import os

import numpy as np
import rasterio
import ray
import torch
import yaml
from ray import serve
from ray.serve.handle import DeploymentHandle
from vllm import AsyncEngineArgs

from geoserve.async_llm_engine_support import AsyncLLMEngineDeployment
from geoserve.geospatial_deployment import GeospatialDeployment
from geoserve.geospatial_preprocessing import PrithviPreprocessor

NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
OFFSET = 0
PERCENTILE = 99

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
    preprocessing = PrithviPreprocessor.bind()
    model = GeospatialDeployment.bind(vllm_deployment=vllm_engine, preprocessor_deployment=preprocessing)

    try:
        handle: DeploymentHandle = serve.run(model, name="geospatial-prithvi-example")


        os.makedirs(output_dir, exist_ok=True)

        with open(config, "r") as f:
            config_dict = yaml.safe_load(f)

        # Loading data ---------------------------------------------------------------------------------
        input_data, temporal_coords, location_coords, meta_data = PrithviPreprocessor.load_example(
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
        print(f"Error: {e}")

    finally:
        # Clean up
        serve.delete("gfinol-geospatial")
        print("Cleaned up resources.")


if __name__ == "__main__":
    ray_address = "ray://localhost:10001"
    geotiff_file = "data/India_900498_S2Hand.tif"
    config = "./model/config.yaml"
    output_dir = "./output"
    runtime_env = {"pip": "requirements.txt"}
    ray.init(address=ray_address, runtime_env=runtime_env)
    main(geotiff_file, config, output_dir)