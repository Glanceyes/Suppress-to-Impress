import math
from typing import List
from PIL import Image, ImageFilter
import cv2
import os
import numpy as np
import torch
import pydensecrf.densecrf as dcrf
from torchvision import transforms
from utils import ptp_utils
from IPython.display import display
from sklearn.decomposition import PCA
from pydensecrf.utils import unary_from_labels
from utils.ptp_utils import AttentionStore, aggregate_attention


def crf_inference_label(img, labels, t=10, n_labels=2, gt_prob=0.7):

    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

    q = d.inference(t)

    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)


def save_attention(prompt: str,
                         timestep: int,
                         attention_store: AttentionStore,
                         tokenizer,
                         save_path: str,
                         indices_to_alter: List[int],
                         res: List[int],
                         from_where: List[str],
                         blur_radius=1.,
                         select: int = 0,
                         resize_to: int = 512,
                         is_cross_attention: bool = True,
                         predicted_image = None,
                         ):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    attention_maps_list = []
    
    res = res if isinstance(res, list) else [res]
    
    for res_val in res:
        attention_maps = aggregate_attention(
            attention_store, 
            res_val, 
            from_where, 
            is_cross_attention, 
            select
        ).detach().cpu()
        
        if not os.path.exists(save_path):
            os.makedirs(save_path) 
        
        attention_path = os.path.join(save_path, f"{'self' if not is_cross_attention else 'cross'}_attention_maps", f"{res_val}")
    
        if not os.path.exists(os.path.join(attention_path)):
            os.makedirs(os.path.join(attention_path))
    
        if is_cross_attention:
            for index in range(len(tokens)):
                attention_maps_numpy = attention_maps[:, :, index]
                if index in indices_to_alter:
                    attention_map_numpy = np.array(attention_maps_numpy)
                    
                    save_cross_attention_path = os.path.join(attention_path, f"{res_val}", f"{decoder(int(tokens[index]))}")
                    
                    if not os.path.exists(save_cross_attention_path):
                        os.makedirs(save_cross_attention_path)
                        
                    save_cross_attention_timestep_path = os.path.join(f"{save_cross_attention_path}", f"{timestep}")
                    np.save(f"{save_cross_attention_timestep_path}.npy", attention_map_numpy)
                    
                    attention_map_numpy = (attention_map_numpy - attention_map_numpy.min()) / (attention_map_numpy.max() - attention_map_numpy.min())
                    
                    zeros = np.zeros_like(attention_map_numpy)
                    
                    attention_map_rgb = Image.fromarray((np.stack([attention_map_numpy] * 3, axis=-1) * 255).astype(np.uint8))
                    attention_map_rgba = Image.fromarray((np.stack([zeros] * 3 + [1. - attention_map_numpy], axis=-1) * 255).astype(np.uint8))
                    
                    attention_map_rgb = attention_map_rgb.filter(ImageFilter.GaussianBlur(blur_radius))
                    attention_map_rgb = transforms.Resize((resize_to, resize_to), interpolation=transforms.InterpolationMode.NEAREST)(attention_map_rgb)
                    
                    attention_map_rgba = attention_map_rgba.filter(ImageFilter.GaussianBlur(blur_radius))
                    attention_map_rgba = transforms.Resize((resize_to, resize_to), interpolation=transforms.InterpolationMode.NEAREST)(attention_map_rgba)
                    
                    attention_map_rgb.save(f"{save_cross_attention_timestep_path}.jpg")
                    attention_map_rgba.save(f"{save_cross_attention_timestep_path}.png")
                    
                    if predicted_image is not None:
                        predicted_image = np.array(predicted_image)
                        
                        # Normalize attn_cam to [0, 1]
                        attention_maps_numpy = (attention_maps_numpy - attention_maps_numpy.min()) / (attention_maps_numpy.max() - attention_maps_numpy.min())
                        
                        attention_cam = cv2.resize(attention_maps_numpy.numpy(), (predicted_image.shape[0], predicted_image.shape[1]))
                        attention_cam = attention_cam[None, :]
                        
                        foreground_cam = np.pad(attention_cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.3)
                        foreground_cam = np.argmax(foreground_cam, axis=0)
                        
                        Image.fromarray((foreground_cam * 255).astype(np.uint8)).save(f"{save_cross_attention_timestep_path}_fg.png")
                        
                        foreground_confidence = crf_inference_label(predicted_image, foreground_cam, t=10, n_labels=2, gt_prob=0.7)
                        
                        background_cam = np.pad(attention_cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.05)
                        background_cam = np.argmax(background_cam, axis=0)
                        
                        background_confidence = crf_inference_label(predicted_image, background_cam, t=10, n_labels=2, gt_prob=0.7)
                        
                        conf = foreground_confidence.copy()
                        conf[foreground_confidence == 0] = 255
                        conf[(background_confidence + foreground_confidence) == 0] = 0
                        
                        attn_map_crf = Image.fromarray((np.stack([conf] * 3, axis=-1)).astype(np.uint8))
                        attn_map_crf.save(f"{save_cross_attention_timestep_path}_crf.jpg")
        else:
            attention_maps_numpy = attention_maps.numpy()
            attention_maps_numpy = attention_maps_numpy.reshape(-1, attention_maps_numpy.shape[-2]*attention_maps_numpy.shape[-3])
            pca = PCA(n_components=3)
            pca.fit(attention_maps_numpy)
            feature_map = pca.transform(attention_maps_numpy)  # N X 3
            
            h = w = int(math.sqrt(feature_map.shape[0]))

            if len(feature_map.shape) == 1:
                feature_map = feature_map.reshape(h, w)
                feature_map_img_min = feature_map.min()
                feature_map_img_max = feature_map.max()
            else:
                feature_map = feature_map.reshape(h, w, 3)
                feature_map_img_min = feature_map.min(axis=(0, 1))
                feature_map_img_max = feature_map.max(axis=(0, 1))


            feature_map = (feature_map - feature_map_img_min) / (feature_map_img_max - feature_map_img_min)
            
            feature_map_rgba = Image.fromarray((feature_map * 255).astype(np.uint8))
            feature_map_rgba = transforms.Resize(512, interpolation=transforms.InterpolationMode.NEAREST)(feature_map_rgba)
            
            save_attention_path = os.path.join(attention_path, f"{timestep}")
            feature_map_rgba.save(f"{save_attention_path}.png")
                


def show_cross_attention(prompt: str,
                         attention_store: AttentionStore,
                         tokenizer,
                         indices_to_alter: List[int],
                         res: int,
                         from_where: List[str],
                         select: int = 0,
                         global_attention: bool = False,
                         orig_image=None):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select, global_attention).detach().cpu()
    images = []

    # show spatial attention for indices of tokens to strengthen
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        if i in indices_to_alter:
            image = show_image_relevance(image, orig_image)
            image = image.astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
            images.append(image)

    ptp_utils.view_images(np.stack(images, axis=0))


def show_self_attention(attention_store: AttentionStore,
                        res: int,
                        from_where: List[str],
                        select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).detach().cpu()
    # attention_maps shape is (res, res, num)
    attn_maps = attention_maps.numpy()
    print(attn_maps.shape)
    attn_maps = attn_maps.reshape(-1, attn_maps.shape[-2]*attn_maps.shape[-3])
    pca = PCA(n_components=3)
    pca.fit(attn_maps)
    feature_map = pca.transform(attn_maps)  # N X 3
    
    h = w = int(math.sqrt(feature_map.shape[0]))

    if len(feature_map.shape) == 1:
        feature_map = feature_map.reshape(h, w)
        feature_map_img_min = feature_map.min()
        feature_map_img_max = feature_map.max()
    else:
        feature_map = feature_map.reshape(h, w, 3)
        feature_map_img_min = feature_map.min(axis=(0, 1))
        feature_map_img_max = feature_map.max(axis=(0, 1))


    feature_map = (feature_map - feature_map_img_min) / (feature_map_img_max - feature_map_img_min)
    
    feature_map_rgba = Image.fromarray((feature_map * 255).astype(np.uint8))
    feature_map_rgba = transforms.Resize(512, interpolation=transforms.InterpolationMode.NEAREST)(feature_map_rgba)
    
    display(feature_map_rgba)
    
    

def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def get_image_grid(images: List[Image.Image]) -> Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = Image.new('RGB', (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image
