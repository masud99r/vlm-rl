import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

import numpy as np
from run_llava import eval_model

# import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from llava.model import *
from llava.model.utils import KeywordsStoppingCriteria


# sentence embedding
from sentence_transformers import SentenceTransformer
sen_model = SentenceTransformer('all-MiniLM-L6-v2')



from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO

import pickle
import pprint

def save_dict_to_file(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def load_image(image_file):
    return image_file

# Model
# disable_torch_init() # ?
model_name = "/home/ubuntu/storage-dev/generative_model/LLaVA/weights/LLaVA-13B-v1-1" # LLaVA-7B-v0, LLaVA-13B-v0, LLaVA-13B-v1-1, replace this with appropiate link
tokenizer = AutoTokenizer.from_pretrained(model_name)

if "mpt" in model_name.lower():
    model = LlavaMPTForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()
else:
    model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()
image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
if mm_use_im_start_end:
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

vision_tower = model.get_model().vision_tower[0]
if vision_tower.device.type == 'meta':
    vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
    model.get_model().vision_tower[0] = vision_tower
else:
    vision_tower.to(device='cuda', dtype=torch.float16)
vision_config = vision_tower.config
vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
vision_config.use_im_start_end = mm_use_im_start_end
if mm_use_im_start_end:
    vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2


def get_text_content(args_image_file, args_query, args_conv_mode):
    qs = args_query
    if mm_use_im_start_end:
        qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
    else:
        qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

    if "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt_multimodal"
    else:
        conv_mode = "multimodal"

    if args_conv_mode is not None and conv_mode != args_conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args_conv_mode, args_conv_mode))
    else:
        args_conv_mode = conv_mode

    conv = conv_templates[args_conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])

    image = load_image(args_image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


# llama2


# run llava
folder_path = "frozen_lake"  #  #"frozen_lake_blue", "frozen_lake"  # Replace with the actual folder path, this folder contains images represent states
query = "Describe the observation"
   
# List all files in the folder
files = os.listdir(folder_path)
print('files', files)
state_dict = {}
text_dict = {}
for file_name in files:
    example_id = os.path.join(folder_path, file_name)

    print(example_id)
    print(file_name)
    # break
    image_file = Image.open(example_id)
    
    
    conv_mode = None
    text_content = get_text_content(image_file, query, conv_mode)
    text_dict[file_name] = text_content

    text_state = sen_model.encode(str(text_content)) # default to numpy array
    state_dict[file_name] = text_state

pprint.pprint(text_dict)

save_dict_to_file(text_dict, folder_path+"_text_dict")
save_dict_to_file(state_dict, folder_path+"_state_dict")


