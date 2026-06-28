import os
import re
import time
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from copy import deepcopy
from safetensors.torch import load_file
from data.data_utils import pil_img2rgb
from modeling.autoencoder import load_ae
from modeling.qwen2 import Qwen2Tokenizer
from data.transforms import ImageTransform
from modeling.bagel.qwen2_navit import NaiveCache
from typing import List, Dict, Optional, Union, Any
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.infer import FusionCharAnnotator, AnnotationRenderer
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vae_transform_gen, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.vae_transform_gen = vae_transform_gen
        self.new_token_ids = new_token_ids
        self.fusion_render = FusionCharAnnotator(
                                        opacity=160,
                                        mask_color=(100, 150, 255),
                                        mask_shape='square',
                                    )
        self.anno_render = AnnotationRenderer(opacity=220,
                                        font_size=42,
                                        mask_color=(100, 150, 255),
                                        text_color=(0, 0, 0),
                                        mask_shape='square',
                                        text_bold=True,
                                        bold_strength=1,
                                        text_stroke_width=0,
                                        text_stroke_color=(0, 0, 0)
                                        )
        
    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )
        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']
        
        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            ## update vit 
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)
        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image_gen(self, image, gen_context, vae=True, vit=True):

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']
        
        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform_gen, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            ## update vit 
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)
        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        
        num_timesteps=50, 
        timestep_shift=3.0
    ):
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
        ) 
        
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
        )

        image = self.decode_image(unpacked_latent[0], image_shape)
        return image

    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)

        vae_dtype = next(self.vae_model.parameters()).dtype
        vae_device = next(self.vae_model.parameters()).device
        latent = latent.to(device=vae_device, dtype=vae_dtype)
        image = self.vae_model.decode(latent)

        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output


class UniHIRInferencer(InterleaveInferencer):


    def merge_ocr_texts(self, original_text, correction_text):
        """
        整合两段OCR文本，根据修正文本更新原始文本
        
        Args:
            original_text: 原始文本（第一段）
            correction_text: 修正文本（第二段）
        
        Returns:
            整合后的新文本
        """
        metadata_match = re.search(r'这张古籍图像出自《([^》]+)》，刻经年代为([^。]+)。', correction_text)
        if metadata_match:
            metadata = f"这张古籍图像出自《{metadata_match.group(1)}》，刻经年代为{metadata_match.group(2)}。"
        else:
            metadata = ""
        
        if '我没有发现需要修改的地方' in correction_text:
            return metadata + original_text if metadata else original_text
        
        box_pattern = r'<\|box_start\|>(\d+),(\d+),(\d+),(\d+),([^<]+)<\|box_end\|>'
        original_boxes = {}
        original_boxes_order = []
        
        for match in re.finditer(box_pattern, original_text):
            x1, y1, x2, y2, char = match.groups()
            coord_key = (int(x1), int(y1), int(x2), int(y2))
            original_boxes[coord_key] = char
            original_boxes_order.append(coord_key)
        
        def coord_distance(coord1, coord2):
            x1_1, y1_1, x2_1, y2_1 = coord1
            x1_2, y1_2, x2_2, y2_2 = coord2
            center1_x = (x1_1 + x2_1) / 2
            center1_y = (y1_1 + y2_1) / 2
            center2_x = (x1_2 + x2_2) / 2
            center2_y = (y1_2 + y2_2) / 2
            return ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
        
        def sort_boxes(boxes_dict):
            if not boxes_dict:
                return []
            
            box_centers = {}
            total_width = 0
            for coord in boxes_dict.keys():
                x1, y1, x2, y2 = coord
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                box_centers[coord] = (center_x, center_y)
                total_width += (x2 - x1)
            
            avg_box_width = total_width / len(boxes_dict) if boxes_dict else 30
            coords_list = list(boxes_dict.keys())
            columns = [[coord] for coord in coords_list]
            column_threshold = avg_box_width * 0.5
            
            merged = True
            while merged:
                merged = False
                new_columns = []
                used = set()
                
                for i, col1 in enumerate(columns):
                    if i in used:
                        continue

                    avg_x1 = sum(box_centers[c][0] for c in col1) / len(col1)
                    merged_col = list(col1)
                    for j, col2 in enumerate(columns[i+1:], start=i+1):
                        if j in used:
                            continue
                        
                        avg_x2 = sum(box_centers[c][0] for c in col2) / len(col2)
                        if abs(avg_x1 - avg_x2) < column_threshold:
                            merged_col.extend(col2)
                            used.add(j)
                            merged = True
                    
                    new_columns.append(merged_col)
                    used.add(i)
                
                columns = new_columns

            for column in columns:
                column.sort(key=lambda c: box_centers[c][1])
            columns.sort(key=lambda col: -sum(box_centers[c][0] for c in col) / len(col))
            
            sorted_coords = []
            for column in columns:
                sorted_coords.extend(column)
            
            return sorted_coords

        corrections = {}
        if '我没有发现破损字符预测错误' not in correction_text and '以下的破损字符预测错误' in correction_text:
            correction_section = re.search(r'以下的破损字符预测错误.*?\n(.*?)(?:\n\n+|$)', correction_text, re.DOTALL)
            if correction_section:
                for match in re.finditer(box_pattern, correction_section.group(1)):
                    x1, y1, x2, y2, new_char = match.groups()
                    coord_key = (int(x1), int(y1), int(x2), int(y2))
                    corrections[coord_key] = new_char
        

        new_boxes = {}
        if '我没有发现破损位置需要新增框' not in correction_text and '以下的破损位置需要新增框' in correction_text:
            new_section = re.search(r'以下的破损位置需要新增框.*?\n(.*?)(?:\n\n+|$)', correction_text, re.DOTALL)
            if new_section:
                for match in re.finditer(box_pattern, new_section.group(1)):
                    x1, y1, x2, y2, char = match.groups()
                    coord_key = (int(x1), int(y1), int(x2), int(y2))
                    new_boxes[coord_key] = char

        delete_coords = set()
        if '我没有发现需要删除的框' not in correction_text and '以下的框需要删除' in correction_text:
            delete_section = re.search(
                r'以下的框需要删除[。\s]*\n(.*?)(?=\n\n+|\n[0-9１-９]）|$)', 
                correction_text, 
                re.DOTALL
            )
            
            if delete_section:
                delete_text = delete_section.group(1).strip()
                
                for match in re.finditer(box_pattern, delete_text):
                    x1, y1, x2, y2, _ = match.groups()
                    coord_key = (int(x1), int(y1), int(x2), int(y2))
                    delete_coords.add(coord_key)
        
        position_corrections = {}
        if '以下框的位置有偏移' in correction_text or '矫正后的结果如下' in correction_text:
            position_section = re.search(
                r'(?:以下框的位置有偏移|矫正后的结果如下)[：:：]*\s*\n(.*?)(?:\n\n+|$)', 
                correction_text, 
                re.DOTALL
            )
            if position_section:
                position_text = position_section.group(1).strip()

                for match in re.finditer(box_pattern, position_text):
                    new_x1, new_y1, new_x2, new_y2, char = match.groups()
                    new_coord = (int(new_x1), int(new_y1), int(new_x2), int(new_y2))

                    best_match = None
                    min_distance = float('inf')
                    
                    for old_coord, old_char in original_boxes.items():
                        if old_char == char and old_coord != new_coord:
                            distance = coord_distance(old_coord, new_coord)
                            if distance < min_distance:
                                min_distance = distance
                                best_match = old_coord
                    
                    if best_match and min_distance < 100:
                        position_corrections[best_match] = new_coord
        
        corrected_boxes = {}
        
        for old_coord in original_boxes_order:
            char = original_boxes[old_coord]
            
            if old_coord in position_corrections:
                new_coord = position_corrections[old_coord]
                corrected_boxes[new_coord] = char
            else:
                corrected_boxes[old_coord] = char
        
        for coord, new_char in corrections.items():
            if coord in corrected_boxes:
                corrected_boxes[coord] = new_char
        
        corrected_boxes.update(new_boxes)
        for old_coord in delete_coords:
            if old_coord in position_corrections:
                new_coord = position_corrections[old_coord]
                corrected_boxes.pop(new_coord, None)
            else:
                corrected_boxes.pop(old_coord, None)
        
        sorted_coords = sort_boxes(corrected_boxes)
        final_text = metadata
        for coord in sorted_coords:
            x1, y1, x2, y2 = coord
            char = corrected_boxes[coord]
            final_text += f"<|box_start|>{x1},{y1},{x2},{y2},{char}<|box_end|>"
        
        return final_text

    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,
        und_only = False,
        gen_only = False,
        gen_first = False,
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        reflection_count=1,
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        output_dict = {}

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)
            
            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    original_img = pil_img2rgb(input_term)
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    input_img = input_term
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")
            
            img = self.gen_image(
                image_shapes, 
                gen_context, 
                cfg_text_precontext=cfg_text_context, 
                cfg_img_precontext=cfg_img_context,
                cfg_text_scale=cfg_text_scale, 
                cfg_img_scale=cfg_img_scale, 
                cfg_interval=cfg_interval, 
                timestep_shift=timestep_shift, 
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
            )
            output_list.append(img)
            output_dict['draft'] = img

            input_term = self.vae_transform.resize_transform(pil_img2rgb(img))
            gen_context = self.update_context_image(input_term, gen_context, vae=False)

            gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
            # output_list.append(gen_text)

        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        print('中间步：' + gen_text.split("\n")[-1])
        output_dict['text_step1'] = gen_text
        output_dict['text_merge'] = gen_text
        print('\n')

        output_dict['tmp_img'] = []
        output_dict['tmp_img_gen'] = []
        output_dict['text_step2'] = ''
        
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            
            input_term = self.vae_transform.resize_transform(pil_img2rgb(img))
            annotations = self.fusion_render.parse_ref_annotations(gen_text)
            annotated_img = self.fusion_render.create_comparison_image(original_img, annotations)
            output_dict['tmp_img'].append(annotated_img)

            input_term = self.vae_transform.resize_transform(annotated_img)

            for i in range(reflection_count):

                start_time = time.time()
                gen_context = self.update_context_image(input_term, gen_context, vae=True)

                pattern = r'<\|box_start\|>(\d+),(\d+),(\d+),(\d+),(.)<\|box_end\|>'
                matches = re.findall(pattern, gen_text)

                reflec_text = ''
                for match in matches:
                    x1, y1, x2, y2, char = match
                    reflec_text += f'<|box_start|>{x1},{y1},{x2},{y2},{char}<|box_end|>'

                prompt = """请对当前页的破损检测结果进行精修。
已给定信息：
1. 破损检测图：在原图上叠加的当前模型预测破损蓝框。
2. 破损定位坐标和预测的字符：每个候选框的 <|box_start|>x1,y1,x2,y2,预测的破损字符<|box_end|>。

你的任务：
- 再次仔细"看图"，结合当前破损检测图、破损定位坐标和预测的字符，对检测结果进行全面精修。请重点完成以下工作：
1）观察破损检测图以及破损定位坐标，分析是否真正破损，若非破损字符，则需要把检测错误的框删除。
2）观察破损检测图以及破损定位坐标，在图像中存在明显破损但当前并没有其破损定位坐标，新增合适大小的框，并补充预测该处的破损字符。
3）观察破损检测图和利用OCR能力得到上下文内容，然后可以先预测这个片段出自于哪里，然后修正当前预测错误的破损字符。
4）观察破损检测图以及破损定位坐标，检查框的位置是否准确，如果框的位置有偏移，需要矫正框的位置。


破损定位坐标和预测的字符："""

                prompt += reflec_text
                cfg_text_context = deepcopy(gen_context)
                gen_context = self.update_context_text(prompt, gen_context)
                cfg_img_context = self.update_context_text(prompt, cfg_img_context)
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                
                if '经过检查，我没有发现需要修改的地方。' in gen_text:
                    output_dict['text_merge'] = reflec_text
                    break

                print(f'第{i+1}次修正\n')
                print(f'{gen_text}\n')
                output_dict['text_step2'] += f'第{i+1}次修正\n'
                output_dict['text_step2'] += gen_text
                output_dict['text_step2'] += '\n\n'

                result = self.merge_ocr_texts(reflec_text, gen_text)
                result = result.replace(' 》', '》')
                output_dict['text_merge'] = result
                output_list.append(result)


                gen_context = self.update_context_text(gen_text, gen_context)
                img_tmp_gen = self.gen_image(
                    image_shapes, 
                    gen_context, 
                    cfg_text_precontext=cfg_text_context, 
                    cfg_img_precontext=cfg_img_context,
                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )
                output_dict['tmp_img_gen'].append(img_tmp_gen)

                print(f'第{i+1}次修正用时：{time.time()-start_time}\n')

                gen_text = result
                annotations = self.fusion_render.parse_ref_annotations(result)
                annotated_img = self.fusion_render.create_comparison_image(original_img, annotations)

                output_dict['tmp_img'].append(annotated_img)

                input_term = self.vae_transform.resize_transform(annotated_img)

                gen_context = self.init_gen_context()
                cfg_text_context = deepcopy(gen_context)
                cfg_img_context = deepcopy(gen_context)

            
            gen_prompt = '请参考图2中的蓝色mask及其中对应的文字内容，对图1中破损区域的字体进行修复。修复时需确保字体风格与图1中其他完好字体保持一致，同时不得影响其他完好字体的内容和样式。'
            annotations = self.anno_render.parse_stepref_annotations(output_dict['text_merge'], original_img.size)
            character_img = self.anno_render.create_comparison_image(original_img, annotations)
            gen_context = self.init_gen_context()
            cfg_text_context = deepcopy(gen_context)
            cfg_img_context = deepcopy(gen_context)

            input_term = self.vae_transform_gen.resize_transform(character_img)
            gen_context = self.update_context_image_gen(input_term, gen_context, vae=True, vit=True)
            image_shapes = input_term.size[::-1]
            cfg_text_context = deepcopy(gen_context)
            gen_context = self.update_context_text(gen_prompt, gen_context)
            cfg_img_context = self.update_context_text(gen_prompt, cfg_img_context)

            gen_text = '我将根据图中破损图片以及破损定位图，对破损图片的破损区域进行字体风格一致的修复，最终输出风格一致的完好图片'
            gen_context = self.update_context_text(gen_text, gen_context)

            img = self.gen_image(
                    image_shapes, 
                    gen_context, 
                    cfg_text_precontext=cfg_text_context, 
                    cfg_img_precontext=cfg_img_context,
                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )

        output_dict['final_image'] = img

        return output_dict
    
    def __call__(
        self, 
        image: Optional[Image.Image] = None, 
        text: Optional[str] = None, 
        **kargs
    ) -> Dict[str, Any]:
        output_dict = {'image': None, 'text': None}

        if image is None and text is None:
            print('Please provide at least one input: either an image or text.')
            return output_dict

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)
        
        output_dict = self.interleave_inference(input_list, **kargs)

        return output_dict

def load_model(model_path, max_mem_per_gpu="80GiB"):

    print(f"Train path: {model_path}")

    # LLM config preparing
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config preparing
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE loading
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    print('VAE model loaded')

    # Bagel config preparing
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=128,
    )
    print('BagelConfig initialized')

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        print('Language model initialized')
        vit_model = SiglipVisionModel(vit_config)
        print('ViT model initialized')
        model = Bagel(language_model, vit_model, config)
        print('Bagel model initialized')
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    print('Model structure prepared')

    # Tokenizer Preparing
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Image Transform Preparing
    vae_transform = ImageTransform(1024, 512, 16, True)
    vae_transform_gen = ImageTransform(2048, 512, 16, True)
    vit_transform = ImageTransform(980, 224, 14)

    print(f'Max memory per GPU: {max_mem_per_gpu}')

    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    print(f'Device map: {device_map}')

    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "model.safetensors"),
        device_map=device_map,
        dtype=torch.bfloat16,
        force_hooks=True,
    )

    model_device = next(model.parameters()).device
    vae_model = vae_model.to(model_device).to(torch.bfloat16) 
    model = model.eval()
    print('Model loaded and set to evaluation mode')
    
    return model, tokenizer, vae_model, vae_transform, vae_transform_gen, vit_transform, new_token_ids

def invert_image(image):
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        image_array = np.array(image)
        inverted = 255 - image_array
        return Image.fromarray(inverted)
    else:
        return 255 - image

def save_results(output_dict, save_path, original_size, img_path):
    output_image = output_dict['final_image'].resize(original_size, Image.LANCZOS)
    text_all = '\n\n\n'.join([
        output_dict['text_step1'],
        output_dict['text_step2'],
        output_dict['text_merge']
    ])
    output_image = invert_image(output_image)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    output_image.save(f'{save_path}/{img_path.split("/")[-1]}')

    with open(f'{save_path}/{img_path.split("/")[-1].replace(".jpg", ".txt")}', 'w') as f:
        f.write(text_all)