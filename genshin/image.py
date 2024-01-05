import os
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from transformers import AutoProcessor, Blip2ForConditionalGeneration

def read_image(img_path):
    image = cv2.imread(img_path)
    assert image is not None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def init_sam(sam_path):
    sam = sam_model_registry["vit_h"](checkpoint=sam_path)
    if torch.cuda.is_available():
        sam.to('cuda:0')
    mask_generator = SamAutomaticMaskGenerator(
        model = sam,
        # crop_n_layers=1
    )
    return mask_generator

def init_blip(blip_model_name):
    dl_model_p = f'Salesforce/{blip_model_name}'
    processor = AutoProcessor.from_pretrained(dl_model_p)
    model = Blip2ForConditionalGeneration.from_pretrained(dl_model_p, torch_dtype=torch.float16)
    if torch.cuda.is_available():
        model.to('cuda:1')
    return model, processor

def generate_mask(mask_generator, img):
    masks = mask_generator.generate(img)
    return masks

def generate_captions(masks, claim, image, processor, model):
    captions = []
    text = f'{claim}. Is this true?'

    inputs = processor(image, text=text, return_tensors="pt").to('cuda:1', torch.float16)
    generated_ids = model.generate(
        **inputs,
        do_sample=False,
        num_beams=3,
        max_new_tokens=100,
        min_length=10, # 1
        repetition_penalty=1.5,
        length_penalty=1.0,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    captions.append(generated_text)

    image_area = image.shape[0] * image.shape[1]

    for mask in tqdm.tqdm(masks):
        # mask = mask['segmentation']
        if mask['area'] < 0.05 * image_area:
            continue
        mask = np.stack([mask['segmentation'] for _ in range(3)], axis=-1)
        inputs = processor(image*mask, text=text, return_tensors="pt").to('cuda:1', torch.float16)
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=3,
            max_new_tokens=100,
            min_length=10, # 1
            repetition_penalty=1.5,
            length_penalty=1.0,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        captions.append(generated_text)

    return captions

def show_anns(anns, save_base, output_prefix):
    plt.figure(figsize=(20,20))
    plt.imshow(image, alpha=0.1)

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    image_area = sorted_anns[0]['segmentation'].shape[0] * sorted_anns[0]['segmentation'].shape[1]
    sorted_anns = list(filter(lambda x: x['area'] >= 0.05 * image_area, sorted_anns))
    if len(sorted_anns) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for i, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [1.0]])
        cv2.imwrite(os.path.join(save_base, f'mask_{output_prefix}_m{i}.png'), m * 255)
        img[m] = color_mask
    ax.imshow(img)

    plt.axis('off')
    plt.savefig(os.path.join(save_base, f'mask_{output_prefix}.jpg'))

def final_decision(captions):
    if len(captions) == 0:
        return False
    if len(captions) == 1:
        return 'YES' in captions[0].upper()
    return 'YES' in captions[0].upper() and any('YES' in x.upper() for x in captions[1:])

if __name__ == '__main__':
    img_path = '/home/01052711/fache/data/jalan-images/images/Y330762520.jpg'
    sam_path = '../segment-anything/checkpoints/sam_vit_h_4b8939.pth'
    blip_model_name = 'blip2-flan-t5-xl'
    save_base = '../results'
    output_prefix = img_path.split('/')[-1].split('.')[0] + '_area0.05'
    claim = 'I can eat melons here.'

    sam_generator = init_sam(sam_path)
    blip_model, blip_processor = init_blip(blip_model_name)

    image = read_image(img_path)
    with torch.no_grad():
        masks = generate_mask(sam_generator, image)
        captions = generate_captions(masks, claim, image, blip_processor, blip_model)
    print('\n'.join(captions))

    print(final_decision(captions))
    
    # show_anns(masks, save_base, output_prefix)
    