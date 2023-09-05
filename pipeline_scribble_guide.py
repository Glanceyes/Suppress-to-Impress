import os
import abc
import warnings
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

import PIL.Image
import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from packaging import version

from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
)

# from torch.cuda.amp import autocast, GradScaler

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.utils import deprecate, logging
from diffusers.utils import PIL_INTERPOLATION, BaseOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim_inverse import DDIMInverseScheduler

from utils.gaussian_smoothing import GaussianSmoothing

from utils.vis_utils import save_attention, show_cross_attention, show_self_attention

logger = logging.get_logger(__name__)
# scaler = GradScaler()

@dataclass
class ScribbleGuideInversionPipelineOutput(BaseOutput, TextualInversionLoaderMixin):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        latents (`torch.FloatTensor`)
            inverted latents tensor
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """
    latents: torch.FloatTensor
    images: Union[List[PIL.Image.Image], np.ndarray]
    


class AttentionControl(abc.ABC):
    def __init__(self):
        self.current_step = 0
        self.current_attention_layer = 0
        self.num_attention_layers = 0
        return
    
    def reset(self):
        self.current_step = 0
        self.current_attention_layer = 0
        return
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_attn_layers(self):
        return 0
    
    @abc.abstractmethod
    def forward(self, attn, is_cross, place_in_unet: str):
        raise NotImplementedError
    
    def __call__(self, attn, is_cross, place_in_unet: str):
        if self.current_attention_layer >= self.num_uncond_attn_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.current_attention_layer += 1
        
        if self.current_attention_layer == self.num_attention_layers + self.num_uncond_attn_layers:
            self.current_attention_layer = 0
            self.current_step += 1
            self.between_steps()
            
    
class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": []
        }
    
    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.gradient_store = {}
        return
    
    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        
    def forward(self, attention, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        
        if attention.shape[1] <= 32 ** 2:
            self.step_store[key].append(attention)
        
        return attention
    
    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()
    
    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def capture_gradients(self, name):
        def hook(grad):
            self.gradient_store[name] = grad
        return hook
        
    
class ScribbleGuideAttentionProcessor:
    def __init__(self, attention_store: AttentionStore, place_in_unet):
        super().__init__()
        self.attention_store = attention_store
        self.place_in_unet = place_in_unet
        return
    
    def __call__(
        self,
        attention: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        timestep=None,
        loss=None
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attention.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        query = attention.to_q(hidden_states)
        
        is_cross = encoder_hidden_states is not None
        
        # Self attention
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attention.norm_cross:
            encoder_hidden_states = attention.norm_encoder_hidden_states(encoder_hidden_states)
            
        key = attention.to_k(encoder_hidden_states)
        value = attention.to_v(encoder_hidden_states)
        
        query = attention.head_to_batch_dim(query)
        key = attention.head_to_batch_dim(key)
        value = attention.head_to_batch_dim(value)
        
        # Multiply query and key to get attention scores
        attention_probs = attention.get_attention_scores(query, key, attention_mask)
        
        self.attention_store(attention_probs, is_cross, self.place_in_unet)
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attention.batch_to_head_dim(hidden_states)
        
        # Linear Projection
        hidden_states = attention.to_out[0](hidden_states)
        
        # Dropout
        hidden_states = attention.to_out[1](hidden_states)
        
        return hidden_states


def merge_attention(attention_store: AttentionStore,
                    resolution: int,
                    which_layer: List[str],
                    is_cross: bool,
                    ) -> torch.Tensor:
    result = []
    
    attention_maps = attention_store.get_average_attention()
    
    num_pixels = resolution ** 2
    for location in which_layer:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                result_map = item.reshape(-1, resolution, resolution, item.shape[-1])
                result.append(result_map)
                
    result = torch.cat(result, dim=0)
    result = result.sum(dim=0) / result.shape[0]
    
    return result
                


def prepare_unet(unet: UNet2DConditionModel, store: AttentionStore):
    attention_processors = {}
    num_attention_layers = 0
    
    for name in unet.attn_processors.keys():
        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        else:
            continue
        
        attention_processors[name] = ScribbleGuideAttentionProcessor(
            attention_store=store, place_in_unet=place_in_unet
        )
        
        num_attention_layers += 1
        
    unet.set_attn_processor(attention_processors)
    store.num_attention_layers = num_attention_layers


class ScribbleGuidePipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = [
        "safety_checker",
        "feature_extractor",
        "caption_generator",
        "caption_processor",
        "inverse_scheduler",
    ]
    
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDPMScheduler, DDIMScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler],
        feature_extractor: CLIPImageProcessor,
        safety_checker: StableDiffusionSafetyChecker,
        inverse_scheduler: DDIMInverseScheduler,
        caption_generator: BlipForConditionalGeneration,
        caption_processor: BlipProcessor,
        requires_safety_checker: bool = False,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            caption_processor=caption_processor,
            caption_generator=caption_generator,
            inverse_scheduler=inverse_scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        
        
    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
            
        # Process prompt if not already embedded
        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
            
            
            # Handle sequence truncation, if necessary
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        
        # Duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # Get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # Duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return text_inputs, prompt_embeds
    
    def preprocess_mask(self,
                        mask: Union[torch.Tensor, PIL.Image.Image], 
                        width: int = 16, 
                        height: int = 16):
        if isinstance(mask, torch.Tensor):
            return mask
        elif isinstance(mask, PIL.Image.Image):
            mask = [mask]

        if isinstance(mask[0], PIL.Image.Image):
            # w, h = mask[0].size
            # w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

            mask = [np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])) for i in mask]
            mask = np.concatenate(mask, axis=0)
            mask = np.array(mask).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
        elif isinstance(mask[0], torch.Tensor):
            mask = torch.cat(mask, dim=0)
            mask = F.interpolate(mask.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False)
            mask = mask.squeeze(0)
        return mask

    def prepare_image_latents(self, image, batch_size, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        if image.shape[1] == 4:
            latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0)
            else:
                latents = self.vae.encode(image).latent_dist.sample(generator)

            latents = self.vae.config.scaling_factor * latents

        if batch_size != latents.shape[0]:
            if batch_size % latents.shape[0] == 0:
                # expand image_latents for batch_size
                deprecation_message = (
                    f"You have passed {batch_size} text prompts (`prompt`), but only {latents.shape[0]} initial"
                    " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                    " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                    " your script to pass as many initial images as text prompts to suppress this warning."
                )
                deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
                additional_latents_per_image = batch_size // latents.shape[0]
                latents = torch.cat([latents] * additional_latents_per_image, dim=0)
            else:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {latents.shape[0]} to {batch_size} text prompts."
                )
        else:
            latents = torch.cat([latents], dim=0)

        return latents
    
    def aggregate_attention(self,
                            attention_maps: torch.Tensor,
                            indices_list: List[int] = None,
                            normalize_eot_token: bool = False
                            ) -> Dict[int, torch.Tensor]:
        last_token_index = -1
        
        if normalize_eot_token:
            prompt = self.prompt
            if isinstance(self.prompt, list):
                prompt = self.prompt[0]
            last_token_index = len(self.tokenizer(prompt)["input_ids"]) - 1
            
        attention_maps = attention_maps[:, :, 1:last_token_index]
        attention_maps *= 100
        attention_maps = torch.nn.functional.softmax(attention_maps, dim=-1)
        
        attention_for_indices = {}
        
        if indices_list is not None:
            indices_list = [index - 1 for index in indices_list]
            for index in indices_list:
                attention_for_indices[index + 1] = attention_maps[:, :, index]
        else:
            for index in range(attention_maps.shape[-1]):
                attention_for_indices[index + 1] = attention_maps[:, :, index]            
    
        return attention_for_indices
    
    def merge_and_aggregate_attention(self,
                                      attention_store: AttentionStore,
                                      resolution: int = 16,
                                      indices_list: List[int] = None,
                                      normalize_eot_token: bool = False,
                                      is_cross: bool = True
                                      ):
        attention_maps = merge_attention(
            attention_store=attention_store,
            resolution=resolution,
            which_layer=["up", "down", "mid"],
            is_cross=is_cross
        )
        
        result = self.aggregate_attention(
            attention_maps=attention_maps,
            indices_list=indices_list,
            normalize_eot_token=normalize_eot_token
        )
        
        return result
    
 
    def contrastive_loss(
        self,
        attention_maps: Dict[int, torch.Tensor],
        indices_list: List[int],
        masks: Dict[int, torch.Tensor],
        contrastive_loss_weight: float = 1.0,
        device: torch.device = torch.device("cpu"),
        reduction: Optional[str] = "mean",
        sigma: float = 0.5,
        kernel_size: int = 3,
        inside_penalty_weight: float = 1.0,
        outside_boost: float = 1.0,
    ) -> torch.Tensor:
        
        loss = 0
        overlap_penalty = 0
        # smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(device)
        
        for index in indices_list:
            mask = masks[index]
            mask_bool = mask.bool()
            
            # other_attention_maps = []
            other_attention_maps_inside = []
            other_attention_maps_outside = []
            max_positions_inside = []
            
            for another_index in attention_maps.keys():
                # other_attention_maps.append(attention_maps[another_index])
                other_attention_maps_inside.append(attention_maps[another_index][mask_bool])
                other_attention_maps_outside.append(attention_maps[another_index][~mask_bool])
                
                # padded_atention_map = F.pad(attention_maps[another_index].unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                # smoothed_attention_map = smoothing(padded_atention_map).squeeze(0).squeeze(0)
                # other_attention_maps_outside.append(smoothed_attention_map[1:-1, 1:-1][~mask_bool])
                # other_attention_maps_outside.append(smoothed_attention_map[~mask_bool])
                
                # Store the position of maximum activation inside the mask
                max_position = torch.argmax(attention_maps[another_index][mask_bool])
                max_positions_inside.append(max_position)
                
            # Exclude the attention map of the current index
            other_attention_maps_inside[index - 1].fill_(0.0)
            
            other_attention_maps_inside = torch.stack(other_attention_maps_inside, dim=0)
            
            # get maximum activation from other attention maps inside mask
            other_inside_result = torch.max(other_attention_maps_inside, dim=0)
            max_other_attention_maps_inside = other_inside_result.values
            
            argmax_other_attention_maps_inside = other_inside_result.indices
            
            max_other_attention_maps_outside = [other_attention_maps_outside[i].max() for i in range(len(other_attention_maps_outside))]
            max_other_attention_maps_outside[index - 1] = 0.0
            
            max_values_outside = torch.tensor(max_other_attention_maps_outside, device=device)
            
            max_values_per_argmax = max_values_outside[argmax_other_attention_maps_inside.flatten()].reshape(argmax_other_attention_maps_inside.shape)
            
            loss += inside_penalty_weight * torch.mean(max_other_attention_maps_inside)
            # loss -= outside_boost * torch.mean(max_values_outside)
            
            loss += torch.mean(max_other_attention_maps_inside - max_values_per_argmax) 
            
            # Check for overlaps in the position of maximum activations and add to the overlap penalty
            # max_positions_inside = torch.tensor(max_positions_inside)
            # unique_positions = torch.unique(max_positions_inside)
            # overlap_penalty += len(max_positions_inside) - len(unique_positions)

        if reduction == "mean":
            loss /= len(indices_list)
            
        return (loss + overlap_penalty) * contrastive_loss_weight
     
    
    def centroid_loss(
        self,
        attention_maps: Dict[int, torch.Tensor],
        indices_list: List[int],
        masks: Dict[int, torch.Tensor],
        centroid_loss_weight: float = 1.0,
        device: torch.device = torch.device("cpu"),
        reduction: Optional[str] = "sum"
    ) -> torch.Tensor:
        loss = 0
        
        for index in indices_list:
            mask = masks[index]
            
            attention_map = attention_maps[index]
            
            y_coords, x_coords = np.meshgrid(np.arange(16), np.arange(16))
            
            y_coords = torch.tensor(y_coords).float().to(device)
            x_coords = torch.tensor(x_coords).float().to(device)
            
            normalized_attention_map = attention_map / attention_map.sum()
            
            attention_map_centroid_x = torch.sum(x_coords * normalized_attention_map)
            attention_map_centroid_y = torch.sum(y_coords * normalized_attention_map)
            
            mask_sum = torch.sum(mask.float())
            
            mask_centroid_x = torch.sum(x_coords * mask.float()) / mask_sum
            mask_centroid_y = torch.sum(y_coords * mask.float()) / mask_sum
            
            loss += torch.abs(attention_map_centroid_x - mask_centroid_x) + torch.abs(attention_map_centroid_y - mask_centroid_y)
        
        if reduction == "mean":
            loss /= len(indices_list)
        
        return loss * centroid_loss_weight
    
    
    def binary_cross_entropy_loss(
        self,
        attention_maps: Dict[int, torch.Tensor],
        indices_list: List[int],
        masks: Dict[int, torch.Tensor],
        binary_cross_entropy_loss_weight: float = 10.0,
        device: torch.device = torch.device("cpu"),
        reduction: Optional[str] = "mean"
    ) -> torch.Tensor:
        loss = 0
        
        for index in indices_list:
            mask = masks[index]
            mask_bool = mask.bool()
            
            attention_map = attention_maps[index]
            
            loss_inside_mask = torch.nn.BCEWithLogitsLoss(
                reduction=reduction
            )(attention_map[mask_bool], mask[mask_bool])
            
            
            loss_outside_mask = torch.nn.BCEWithLogitsLoss(
                reduction=reduction
            )(attention_map[~mask_bool], mask[~mask_bool])
            
            inside_area = torch.sum(mask_bool).float()
            outside_area = torch.sum(~mask_bool).float()
            
            total_area = inside_area + outside_area
            
            inside_weight = inside_area / total_area
            outside_weight = outside_area / total_area
            
            loss += (loss_inside_mask * inside_weight + loss_outside_mask * outside_weight) 
        
        return loss * binary_cross_entropy_loss_weight
    
                  
    def compute_loss(
        self,
        attention_maps: Dict[int, torch.Tensor],
        indices_list: List[int],
        masks: Dict[int, torch.Tensor],
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        loss = 0
        
        loss += self.binary_cross_entropy_loss(
            attention_maps=attention_maps,
            indices_list=indices_list,
            masks=masks,
            device=device,
        )
        
        loss += self.contrastive_loss(
            attention_maps=attention_maps,
            indices_list=indices_list,
            masks=masks,
            device=device,
        )
        
        return loss
    
    @torch.no_grad()
    def generate_caption(self, images):
        """Generates caption for a given image."""
        text = "a photography of"

        prev_device = self.caption_generator.device

        device = self._execution_device
        inputs = self.caption_processor(images, text, return_tensors="pt").to(
            device=device, dtype=self.caption_generator.dtype
        )
        self.caption_generator.to(device)
        outputs = self.caption_generator.generate(**inputs, max_new_tokens=128)

        # offload caption generator
        self.caption_generator.to(prev_device)

        caption = self.caption_processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return caption
    
    
    @staticmethod
    def update_latent(
        latents: torch.Tensor,
        loss: torch.Tensor,
        step_size: float
    ) -> torch.Tensor:
        gradient = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * gradient
        return latents
    
    
    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            attention_store: AttentionStore,
            indices_list: List[int],
            token_masks: Union[
                List[torch.Tensor],
                List[PIL.Image.Image],
                List[np.ndarray],
            ] = None,
            attention_resolution: int = 16,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: Optional[float] = 0.0,
            seed: Optional[int] = 21,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            run_standard: bool = False,
            scale_factor: int = 10,
            scale_range: Tuple[float, float] = (1., 0.5),
            output_path: str = 'runs',
            save_cross_attention: bool = True,
            save_self_attention: bool = True,
        ):
        
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, 
            negative_prompt, prompt_embeds, negative_prompt_embeds
        )
        
        indices_list.sort()
        
        # 2. Define call parameters
        self.prompt = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
            
        device = self._execution_device
        # `guidance_scale = 1` corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # 3. Encode input prompt
        text_inputs, prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        masks = {}
        
        # 4.1. Preprocess Token Masks
        if token_masks is not None:
            for index, mask in zip(indices_list, token_masks):
                token_mask = self.preprocess_mask(mask, attention_resolution, attention_resolution)
                masks[index] = token_mask.to(device)
                
        # 5. Prepare latent variables
        if latents is None:
            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
        
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))
        
        if attention_store is None:
            attention_store = AttentionStore()
        
        prepare_unet(self.unet, attention_store)
        
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, timestep in enumerate(timesteps):
                
                if not run_standard:
                    with torch.enable_grad():
                        latents = latents.clone().detach().requires_grad_(True)
                        refinement_num_steps = max(1, 25 - i)
                        cross_attention_maps = None
                    
                        for j in range(1, refinement_num_steps):
                            # Forward pass of denoising with text conditioning
                            if token_masks is not None:
                                # with autocast():
                                if cross_attention_maps is None:
                                    noise_pred_text = self.unet(
                                        latents, 
                                        timestep,
                                        encoder_hidden_states=prompt_embeds[1].unsqueeze(0),
                                        cross_attention_kwargs=cross_attention_kwargs
                                    ).sample
                                    self.unet.zero_grad()
                                    cross_attention_maps = self.merge_and_aggregate_attention(
                                        attention_store=attention_store,
                                        resolution=attention_resolution,
                                        # indices_list=indices_list,
                                        normalize_eot_token=False,
                                        is_cross = True
                                    )
                                    
                                hook_handles = []
                                for key, attention_map_tensor in cross_attention_maps.items():
                                    attention_map_with_grad = attention_map_tensor.clone().requires_grad_(True)
                                    handle = attention_map_with_grad.register_hook(attention_store.capture_gradients(key))
                                    hook_handles.append(handle)
                                    cross_attention_maps[key] = attention_map_with_grad
                                
                                loss = self.compute_loss(
                                    attention_maps=cross_attention_maps,
                                    indices_list=indices_list,
                                    masks=masks,
                                    device=device,
                                )
                                
                                if loss != 0:
                                    loss.backward(retain_graph=True)
                                    
                                    # attention_store.gradient_store = {}
                                    if j < refinement_num_steps - 1:
                                        for key, value in cross_attention_maps.items():
                                            gradient = attention_store.gradient_store[key]
                                            cross_attention_maps[key] = value + scale_factor * np.sqrt(scale_range[i]) * gradient
                            
                                    else:
                                        latents = self.update_latent(
                                            latents=latents,
                                            loss=loss,
                                            step_size=scale_factor * np.sqrt(scale_range[i])
                                        )
                                            
                                for handle in hook_handles:
                                    handle.remove()
                            
                    
                latents_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latents_model_input = self.scheduler.scale_model_input(latents_model_input, timestep)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latents_model_input,
                        timestep,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
                    
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        
                    # compute the previous noisy sample x_t -> x_t-1
                    step_backward_result = self.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs)
                    latents = step_backward_result.prev_sample
                    predicted_sample = step_backward_result.pred_original_sample
                    
                    predicted_image = self.decode_latents(predicted_sample)
                    
                    if output_type == "pil" and not run_standard:
                        predicted_image = self.numpy_to_pil(predicted_image)
                        
                        for caption, image in zip(prompt, predicted_image):
                            caption_name = caption.replace(" ", "_")
                            save_output_path = os.path.join(output_path, caption_name, str(seed))
                        
                            if save_cross_attention:
                                save_attention(
                                    caption,
                                    timestep,
                                    attention_store,
                                    self.tokenizer,
                                    save_output_path,
                                    indices_list,
                                    res=attention_resolution,
                                    from_where=("up", "down", "mid"),
                                    is_cross_attention=True,
                                    predicted_image=image
                                )
                            
                            if save_self_attention:
                                save_attention(
                                    caption,
                                    timestep,
                                    attention_store,
                                    self.tokenizer,
                                    save_output_path,
                                    indices_list,
                                    res=attention_resolution,
                                    from_where=("up", "down", "mid"),
                                    is_cross_attention=False,
                                    predicted_image=image
                                )
                            
                            save_image_path = os.path.join(save_output_path, "samples", f"{timestep}.png")
                            
                            if not os.path.exists(os.path.dirname(save_image_path)):
                                os.makedirs(os.path.dirname(save_image_path))
                            
                            image.save(save_image_path)

                    
                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, timestep, latents)
                
        # 8. Post-processing
        sample_image = self.decode_latents(latents)
        # sample_image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        
        has_nsfw_concept = False
        
        # do_denormalize = [True] * sample_image.shape[0]
        # sample_image = self.image_processor.postprocess(sample_image, output_type=output_type, do_denormalize=do_denormalize)
        
        if output_type == "pil":
            sample_image = self.numpy_to_pil(sample_image)
            
        if not return_dict:
            return (sample_image, has_nsfw_concept)
        
        return StableDiffusionPipelineOutput(images=sample_image, nsfw_content_detected=has_nsfw_concept)
    
    
    def get_epsilon(self, model_output: torch.Tensor, sample: torch.Tensor, timestep: int):
        pred_type = self.inverse_scheduler.config.prediction_type
        alpha_prod_t = self.inverse_scheduler.alphas_cumprod[timestep]

        beta_prod_t = 1 - alpha_prod_t

        if pred_type == "epsilon":
            return model_output
        elif pred_type == "sample":
            return (sample - alpha_prod_t ** (0.5) * model_output) / beta_prod_t ** (0.5)
        elif pred_type == "v_prediction":
            return (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {pred_type} must be one of `epsilon`, `sample`, or `v_prediction`"
            )
    
    
    def auto_correlation_loss(self, hidden_states, generator=None):
        reg_loss = 0.0
        for i in range(hidden_states.shape[0]):
            for j in range(hidden_states.shape[1]):
                noise = hidden_states[i : i + 1, j : j + 1, :, :]
                while True:
                    roll_amount = torch.randint(noise.shape[2] // 2, (1,), generator=generator).item()
                    reg_loss += (noise * torch.roll(noise, shifts=roll_amount, dims=2)).mean() ** 2
                    reg_loss += (noise * torch.roll(noise, shifts=roll_amount, dims=3)).mean() ** 2

                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
        return reg_loss
    
    
    def kl_divergence_loss(self, hidden_states):
        mean = hidden_states.mean()
        var = hidden_states.var()
        return var + mean**2 - 1 - torch.log(var + 1e-7)
    
    
    @torch.no_grad()
    def invert(
        self,
        prompt: Optional[str] = None,
        image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_store: AttentionStore = None,
        auto_correlation_weight: float = 20.0,
        kl_divergence_weight: float = 20.0,
        num_reg_steps: int = 5,
        num_auto_correlation_rolls: int = 5,
    ):
        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
            
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        
        if attention_store is None:
            attention_store = AttentionStore()
        
        prepare_unet(self.unet, attention_store)
        
        # 3. Preprocess image
        image = self.image_processor.preprocess(image)
        
        # 4. Prepare latent variables
        latents = self.prepare_image_latents(
            image,
            batch_size,
            self.vae.dtype,
            device,
            generator
        )
        
        # 5. Encode input prompt
        text_inputs , prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
        )
        
        # 6. Prepare timesteps
        self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.inverse_scheduler.timesteps
        
        # 7. Denoising loop where we obtain the cross-attention maps.
        num_warmup_steps = len(timesteps) - num_inference_steps * self.inverse_scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, timestep in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.inverse_scheduler.scale_model_input(latent_model_input, timestep)
                
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                # regularization of the noise prediction
                with torch.enable_grad():
                    for _ in range(num_reg_steps):
                        if auto_correlation_weight > 0:
                            for _ in range(num_auto_correlation_rolls):
                                var = torch.autograd.Variable(noise_pred.detach().clone(), requires_grad=True)

                                # Derive epsilon from model output before regularizing to IID standard normal
                                var_epsilon = self.get_epsilon(var, latent_model_input.detach(), timestep)

                                l_ac = self.auto_correlation_loss(var_epsilon, generator=generator)
                                l_ac.backward()

                                grad = var.grad.detach() / num_auto_correlation_rolls
                                noise_pred = noise_pred - auto_correlation_weight * grad


                        if kl_divergence_weight > 0:
                            var = torch.autograd.Variable(noise_pred.detach().clone(), requires_grad=True)

                            # Derive epsilon from model output before regularizing to IID standard normal
                            var_epsilon = self.get_epsilon(var, latent_model_input.detach(), timestep)

                            l_kld = self.kl_divergence_loss(var_epsilon)
                            l_kld.backward()

                            grad = var.grad.detach()
                            noise_pred = noise_pred - kl_divergence_weight * grad
                            
                        noise_pred = noise_pred.detach()
                        
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.inverse_scheduler.step(noise_pred, timestep, latents).prev_sample
                
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.inverse_scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, timestep, latents)
                        
        inverted_latents = latents.detach().clone()
        
        # 8. Post-processing
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
        
        if not return_dict:
            return (inverted_latents, image)
        
        return ScribbleGuideInversionPipelineOutput(
            latents=inverted_latents,
            images=image,
        )