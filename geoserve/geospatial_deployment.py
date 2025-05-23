import uuid

import torch
from einops import rearrange
from ray import serve
from ray.serve.handle import DeploymentHandle
from torch import Tensor
from vllm import SamplingParams, PoolingRequestOutput


@serve.deployment
class GeospatialDeployment():

    def __init__(self, vllm_deployment: DeploymentHandle, preprocessor_deployment: DeploymentHandle):
        self.vllmDeployment = vllm_deployment.options(stream=True)
        self.preprocessorDeployment = preprocessor_deployment

        self.id = str(uuid.uuid4())

    async def encode(self, geotiff_path: str) -> Tensor:
        preprocessor_response = await self.preprocessorDeployment.apply.remote(geotiff_path)

        location_coords = preprocessor_response["location_coords"]
        pixel_values_chunks = preprocessor_response["pixel_values_chunks"]

        img_size = preprocessor_response["img_size"]
        h1 = preprocessor_response["h1"]
        w1 = preprocessor_response["w1"]
        original_h = preprocessor_response["original_h"]
        original_w = preprocessor_response["original_w"]

        pred_futures = []
        for pixel_values in pixel_values_chunks:
            mm_data = {
                "pixel_values": torch.empty(0) if pixel_values is None else pixel_values,
                "location_coords": torch.empty(0) if location_coords is None else location_coords,
                "temporal_coords": torch.empty(0),
            }

            prompt = {
                "prompt_token_ids": [1],
                "multi_modal_data": mm_data
            }

            pred = self.vllmDeployment.encode.remote(prompt, SamplingParams(temperature=0.0), uuid.uuid4())
            pred_futures.append(pred)

        pred_imgs = []
        for f in pred_futures:
            request_output: PoolingRequestOutput = await f.__anext__()
            pred = request_output.outputs.data

            y_hat = pred.argmax(dim=1)
            y_hat = torch.nn.functional.interpolate(y_hat.unsqueeze(1).float(), size=img_size, mode="nearest")
            pred_imgs.append(y_hat)

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

        return pred_imgs