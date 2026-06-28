import os
import re
import json
import glob
from tqdm import tqdm
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

class FusionCharAnnotator:
    def __init__(self, 
                 opacity=128, 
                 mask_color=(255, 0, 0),  # mask颜色 RGB
                 mask_shape='square'):  # 'circle' 或 'square'
        """
        Args:
            opacity: mask透明度 (0-255, 0完全透明，255完全不透明)
            mask_color: mask颜色，RGB元组，如 (255, 0, 0) 为红色
            mask_shape: mask形状，'circle' 或 'square'
        """
        self.opacity = opacity
        self.mask_color = mask_color
        self.mask_shape = mask_shape

    def parse_ref_annotations(self, output_text):
        annotation_pattern = r'<\|box_start\|>(\d+),(\d+),(\d+),(\d+),(.)<\|box_end\|>'
        matches = re.findall(annotation_pattern, output_text)
        
        annotations = []
        for match in matches:
            x1, y1, x2, y2 = int(match[0]), int(match[1]), int(match[2]), int(match[3])
            annotations.append((x1, y1, x2, y2, ''))
        
        return annotations
    
    def create_comparison_image(self, original_img, annotations):

        image = original_img.copy()
        image = image.convert('RGBA')
        width, height = image.size
        
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        for i, (x1, y1, x2, y2, char) in enumerate(annotations):
            actual_x1 = int(x1 * width / 1000)
            actual_y1 = int(y1 * height / 1000)
            actual_x2 = int(x2 * width / 1000)
            actual_y2 = int(y2 * height / 1000)

            actual_x1 -= 2
            actual_y1 -= 2
            actual_x2 += 2
            actual_y2 += 2

            mask_color_with_alpha = (*self.mask_color, self.opacity)
            outline_color = (255, 255, 255, self.opacity)
            
            if self.mask_shape == 'circle':
                draw.ellipse([
                    actual_x1, actual_y1,
                    actual_x2, actual_y2
                ], fill=mask_color_with_alpha, outline=outline_color, width=2)
            elif self.mask_shape == 'square':
                draw.rectangle([
                    actual_x1, actual_y1,
                    actual_x2, actual_y2
                ], fill=mask_color_with_alpha, outline=outline_color, width=2)

        result = Image.alpha_composite(image, overlay)
        
        return result.convert('RGB')

class AnnotationRenderer:
    def __init__(self, 
                 opacity=128, 
                 font_size=20,
                 mask_color=(255, 0, 0),  # mask颜色 RGB
                 text_color=(255, 255, 255),  # 文字颜色 RGB
                 mask_shape='square',  # 'circle' 或 'square'
                 text_bold=False,  # 文字是否加粗
                 bold_strength=1,  # 加粗强度 (1-3)
                 text_stroke_width=0,  # 文字描边宽度
                 text_stroke_color=(0, 0, 0)):  # 文字描边颜色
        """
        初始化渲染器
        Args:
            opacity: 透明度 (0-255, 0完全透明，255完全不透明)
            font_size: 字体大小
            mask_color: mask颜色，RGB元组，如 (255, 0, 0) 为红色
            text_color: 文字颜色，RGB元组
            mask_shape: mask形状，'circle' 或 'square'
            text_bold: 文字是否加粗
            bold_strength: 加粗强度，1-3（1最轻，3最重）
            text_stroke_width: 文字描边宽度
            text_stroke_color: 文字描边颜色
        """
        self.opacity = opacity
        self.font_size = font_size
        self.mask_color = mask_color
        self.text_color = text_color
        self.mask_shape = mask_shape
        self.text_bold = text_bold
        self.bold_strength = max(1, min(3, bold_strength))
        self.text_stroke_width = text_stroke_width
        self.text_stroke_color = text_stroke_color
        
        font_list = [
            "./font/KaiXinSongA.ttf",
            "./font/KaiXinSongB.ttf",
            "./font/TH-Tshyn-P0.ttf", 
            "./font/TH-Tshyn-P1.ttf", 
            "./font/TH-Tshyn-P16.ttf", 
            "./font/TH-Tshyn-P2.ttf", 
        ]
        
        self.ImageFont_fonts = []
        self.TTFont_fonts = []
        self.font_char_cache = {}
        
        try:
            for font_path in font_list:
                if os.path.exists(font_path):
                    try:
                        pil_font = ImageFont.truetype(font_path, self.font_size)
                        self.ImageFont_fonts.append(pil_font)
                        tt_font = TTFont(font_path)
                        self.TTFont_fonts.append(tt_font)
                        print(f"成功加载字体: {os.path.basename(font_path)}")
                    except Exception as e:
                        print(f"加载字体失败 {font_path}: {e}")
            
            if not self.ImageFont_fonts:
                default_font = ImageFont.load_default()
                self.ImageFont_fonts.append(default_font)
                self.TTFont_fonts.append(None)
                print("警告: 未找到任何字体文件，使用默认字体")
            else:
                print(f"总共加载了 {len(self.ImageFont_fonts)} 个字体文件")
                
        except Exception as e:
            print(f"字体初始化失败: {e}")
            default_font = ImageFont.load_default()
            self.ImageFont_fonts.append(default_font)
            self.TTFont_fonts.append(None)
    
    def is_char_in_font(self, TTFont_font, char):
        if TTFont_font is None:  # 默认字体的情况
            return True
        try:
            cmap = TTFont_font['cmap']
            for subtable in cmap.tables:
                if ord(char) in subtable.cmap:
                    return True
            return False
        except Exception as e:
            print(f"检查字符 '{char}' 时出错: {e}")
            return False

    def find_font_for_char(self, char):
        if char in self.font_char_cache:
            return self.font_char_cache[char]
        
        for i, (ImageFont_font, TTFont_font) in enumerate(zip(self.ImageFont_fonts, self.TTFont_fonts)):
            if self.is_char_in_font(TTFont_font, char):
                self.font_char_cache[char] = ImageFont_font
                return ImageFont_font
            if i == len(self.ImageFont_fonts) - 1:
                print(f"No font can render the character {char}")
                fallback_font = self.ImageFont_fonts[0]
                self.font_char_cache[char] = fallback_font
                return fallback_font
        
        fallback_font = self.ImageFont_fonts[0]
        self.font_char_cache[char] = fallback_font
        return fallback_font

    def adjust_overlapping_boxes(self, annotations):
        if len(annotations) <= 1:
            return annotations
        
        def calculate_iou(box1, box2):
            x1_1, y1_1, x2_1, y2_1 = box1[:4]
            x1_2, y1_2, x2_2, y2_2 = box2[:4]
            
            x_left = max(x1_1, x1_2)
            y_top = max(y1_1, y1_2)
            x_right = min(x2_1, x2_2)
            y_bottom = min(y2_1, y2_2)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            
            intersection = (x_right - x_left) * (y_bottom - y_top)
            
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0
        
        sorted_annotations = sorted(annotations, key=lambda x: (x[1], x[0]))
        adjusted = list(sorted_annotations)
        
        for i in range(len(adjusted)):
            for j in range(i + 1, len(adjusted)):
                box_i = adjusted[i]
                box_j = adjusted[j]
                
                iou = calculate_iou(box_i, box_j)
                
                if iou > 0:
                    x1_i, y1_i, x2_i, y2_i, char_i = box_i
                    x1_j, y1_j, x2_j, y2_j, char_j = box_j

                    overlap_left = max(x1_i, x1_j)
                    overlap_right = min(x2_i, x2_j)
                    overlap_top = max(y1_i, y1_j)
                    overlap_bottom = min(y2_i, y2_j)
                    
                    overlap_width = overlap_right - overlap_left
                    overlap_height = overlap_bottom - overlap_top
                    
                    if overlap_width < overlap_height:
                        if (x1_i + x2_i) / 2 < (x1_j + x2_j) / 2:
                            mid = (overlap_left + overlap_right) // 2
                            adjusted[i] = (x1_i, y1_i, mid, y2_i, char_i)
                            adjusted[j] = (mid, y1_j, x2_j, y2_j, char_j)
                        else:
                            mid = (overlap_left + overlap_right) // 2
                            adjusted[j] = (x1_j, y1_j, mid, y2_j, char_j)
                            adjusted[i] = (mid, y1_i, x2_i, y2_i, char_i)
                    else:
                        if (y1_i + y2_i) / 2 < (y1_j + y2_j) / 2:
                            mid = (overlap_top + overlap_bottom) // 2
                            adjusted[i] = (x1_i, y1_i, x2_i, mid, char_i)
                            adjusted[j] = (x1_j, mid, x2_j, y2_j, char_j)
                        else:
                            mid = (overlap_top + overlap_bottom) // 2
                            adjusted[j] = (x1_j, y1_j, x2_j, mid, char_j)
                            adjusted[i] = (x1_i, mid, x2_i, y2_i, char_i)
        
        return adjusted

    def parse_stepref_annotations(self, output_text, image_size):
        width, height = image_size
        
        annotation_pattern = r'<\|box_start\|>(\d+),(\d+),(\d+),(\d+),(.)<\|box_end\|>'
        matches = re.findall(annotation_pattern, output_text)
        
        annotations = []
        for match in matches:
            x1, y1, x2, y2, char = int(match[0]), int(match[1]), int(match[2]), int(match[3]), match[4]
            
            actual_x1 = int(x1 * width / 1000)
            actual_y1 = int(y1 * height / 1000)
            actual_x2 = int(x2 * width / 1000)
            actual_y2 = int(y2 * height / 1000)
            
            actual_x1 -= 2
            actual_y1 -= 2
            actual_x2 += 2
            actual_y2 += 2
            
            actual_x1 = max(0, actual_x1)
            actual_y1 = max(0, actual_y1)
            actual_x2 = min(width, actual_x2)
            actual_y2 = min(height, actual_y2)
            
            annotations.append((actual_x1, actual_y1, actual_x2, actual_y2, char))
        annotations = self.adjust_overlapping_boxes(annotations.copy())
        
        return annotations
    
    
    def render_char_with_size(self, char, width, height, font_base):

        try:
            if width <= 0 or height <= 0:
                print(f"无效的框尺寸: width={width}, height={height}")
                return font_base
            
            initial_size = max(min(width, height) * 0.1, 10)
            font_size = int(initial_size)
            
            if font_size <= 0:
                font_size = 12 
            
            font = ImageFont.truetype(font_base.path, font_size) if hasattr(font_base, 'path') else font_base

            temp_img = Image.new('RGBA', (width*2, height*2), (255, 255, 255, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            
            bbox = temp_draw.textbbox((0, 0), char, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            if text_width > 0 and text_height > 0:
                scale_w = width * 0.95 / text_width
                scale_h = height * 0.95 / text_height
                scale = min(scale_w, scale_h)
                
                adjusted_font_size = max(8, int(font_size * scale))
                if adjusted_font_size <= 0:
                    adjusted_font_size = 8
                
                font = ImageFont.truetype(font_base.path, adjusted_font_size) if hasattr(font_base, 'path') else font_base
            
            return font
            
        except Exception as e:
            print(f"调整字体大小失败: {e}, 使用默认字体")
            return font_base
    
    def draw_text_with_effects(self, draw, position, text, font, fill_color, stroke_width, stroke_color):
        x, y = position
        
        if stroke_width > 0:
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), text, font=font, fill=stroke_color)

        if self.text_bold:
            offsets = []
            
            if self.bold_strength == 1:
                offsets = [(1, 0)]
            elif self.bold_strength == 2:
                offsets = [(1, 0), (0, 1)]
            else:
                offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]

            for dx, dy in offsets:
                draw.text((x + dx, y + dy), text, font=font, fill=fill_color)

        draw.text(position, text, font=font, fill=fill_color)
    
    def create_comparison_image(self, original_img, annotations):

        image = original_img.copy()
        image = image.convert('RGBA')
        img_width, img_height = image.size
        
        overlay = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        for i, (x1, y1, x2, y2, char) in enumerate(annotations):

            box_width = x2 - x1
            box_height = y2 - y1
            
            if box_width <= 0 or box_height <= 0:
                print(f"  跳过无效边界框: {x1},{y1},{x2},{y2}")
                continue
            
            mask_color_with_alpha = (*self.mask_color, self.opacity)
            outline_color = (255, 255, 255, self.opacity)
            
            if self.mask_shape == 'circle':
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                radius = min(box_width, box_height) // 2
                draw.ellipse([
                    center_x - radius, center_y - radius,
                    center_x + radius, center_y + radius
                ], fill=mask_color_with_alpha, outline=outline_color, width=2)
            elif self.mask_shape == 'square':
                draw.rectangle([x1, y1, x2, y2], 
                             fill=mask_color_with_alpha, outline=outline_color, width=2)
            
            base_font = self.find_font_for_char(char)
            adjusted_font = self.render_char_with_size(char, box_width, box_height, base_font)
            
            try:
                bbox = draw.textbbox((0, 0), char, font=adjusted_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                try:
                    text_width, text_height = draw.textsize(char, font=adjusted_font)
                except:
                    text_width, text_height = box_width // 2, box_height // 2
            
            text_x = x1 + (box_width - text_width) // 2
            text_y = y1 + (box_height - text_height) // 2
            
            text_alpha = 200
            text_color_with_alpha = (*self.text_color, text_alpha)
            stroke_color_with_alpha = (*self.text_stroke_color, text_alpha)
            
            self.draw_text_with_effects(
                draw, (text_x, text_y), char, adjusted_font,
                text_color_with_alpha, self.text_stroke_width, stroke_color_with_alpha
            )
        
        result = Image.alpha_composite(image, overlay)

        return result.convert('RGB')
