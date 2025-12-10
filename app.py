import os
import re
import json
import time
import unicodedata
import gc
from io import BytesIO
from typing import Iterable, Tuple, Optional, List, Dict, Any

import gradio as gr
import numpy as np
import torch
import spaces
from PIL import Image, ImageDraw, ImageFont

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoTokenizer, 
    AutoModelForVision2Seq
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from qwen_vl_utils import process_vision_info

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

print("🔄 Loading Fara-7B...")
MODEL_ID_V = "microsoft/Fara-7B" 
try:
    processor_v = AutoProcessor.from_pretrained(MODEL_ID_V, trust_remote_code=True)
    model_v = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID_V,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(device).eval()
except Exception as e:
    print(f"Failed to load Fara: {e}")
    model_v = None
    processor_v = None

print("🔄 Loading UI-TARS-1.5-7B...")
MODEL_ID_X = "ByteDance-Seed/UI-TARS-1.5-7B"
try:
    processor_x = AutoProcessor.from_pretrained(MODEL_ID_X, trust_remote_code=True, use_fast=False)
    model_x = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID_X,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()
except Exception as e:
    print(f"Failed to load UI-TARS: {e}")
    model_x = None
    processor_x = None

print("🔄 Loading Holo2-4B...")
MODEL_ID_H = "Hcompany/Holo2-4B" 
try:
    processor_h = AutoProcessor.from_pretrained(MODEL_ID_H, trust_remote_code=True)
    model_h = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID_H,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()
except Exception as e:
    print(f"Failed to load Holo2: {e}")
    model_h = None
    processor_h = None

print("🔄 Loading ActIO-UI-7B...")
MODEL_ID_ACT = "Uniphore/actio-ui-7b-rlvr"
try:
    # ActIO usually relies on Qwen2VL architecture structure
    processor_act = AutoProcessor.from_pretrained(MODEL_ID_ACT, trust_remote_code=True)
    model_act = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID_ACT,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=None # We will move to device manually to control memory
    ).to(device).eval()
except Exception as e:
    print(f"Failed to load ActIO-UI: {e}")
    model_act = None
    processor_act = None

print("✅ Models loading sequence complete.")

def array_to_image(image_array: np.ndarray) -> Image.Image:
    if image_array is None: raise ValueError("No image provided.")
    return Image.fromarray(np.uint8(image_array))

def get_image_proc_params(processor) -> Dict[str, int]:
    ip = getattr(processor, "image_processor", None)
    
    default_min = 256 * 256
    default_max = 1280 * 1280

    patch_size = getattr(ip, "patch_size", 14)
    merge_size = getattr(ip, "merge_size", 2)
    min_pixels = getattr(ip, "min_pixels", default_min)
    max_pixels = getattr(ip, "max_pixels", default_max)

    # Holo2/Qwen specific sizing sometimes in 'size' dict
    size_config = getattr(ip, "size", {})
    if isinstance(size_config, dict):
        if "shortest_edge" in size_config:
            min_pixels = size_config["shortest_edge"]
        if "longest_edge" in size_config:
            max_pixels = size_config["longest_edge"]

    if min_pixels is None: min_pixels = default_min
    if max_pixels is None: max_pixels = default_max

    return {
        "patch_size": patch_size,
        "merge_size": merge_size,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
    }

def apply_chat_template_compat(processor, messages: List[Dict[str, Any]], thinking: bool = True) -> str:
    # Holo2 specific: allows turning thinking off in template
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, thinking=thinking)
        except TypeError:
            # Fallback for processors that don't support 'thinking' kwarg
            return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    raise AttributeError("Could not apply chat template.")

def trim_generated(generated_ids, inputs):
    in_ids = getattr(inputs, "input_ids", None)
    if in_ids is None and isinstance(inputs, dict):
        in_ids = inputs.get("input_ids", None)
    if in_ids is None:
        return generated_ids
    return [out_ids[len(in_seq):] for in_seq, out_ids in zip(in_ids, generated_ids)]

def get_fara_prompt(task, image):
    OS_SYSTEM_PROMPT = """You are a GUI agent. You are given a task and a screenshot of the current status.
    You need to generate the next action to complete the task.
    Output your action inside a <tool_call> block using JSON format.
    Include "coordinate": [x, y] in pixels for interactions.
    Examples:
    <tool_call>{"name": "User", "arguments": {"action": "click", "coordinate": [400, 300]}}</tool_call>
    <tool_call>{"name": "User", "arguments": {"action": "type", "coordinate": [100, 200], "text": "hello"}}</tool_call>
    """
    return [
        {"role": "system", "content": [{"type": "text", "text": OS_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": f"Instruction: {task}"}]},
    ]

def get_localization_prompt(task, image):
    guidelines = (
        "Localize an element on the GUI image according to my instructions and "
        "output a click position as Click(x, y) with x num pixels from the left edge "
        "and y num pixels from the top edge."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{guidelines}\n{task}"}
            ],
        }
    ]

def get_holo2_prompt(task, image):
    schema_str = '{"properties": {"x": {"description": "The x coordinate, normalized between 0 and 1000.", "ge": 0, "le": 1000, "title": "X", "type": "integer"}, "y": {"description": "The y coordinate, normalized between 0 and 1000.", "ge": 0, "le": 1000, "title": "Y", "type": "integer"}}, "required": ["x", "y"], "title": "ClickCoordinates", "type": "object"}'
    
    prompt = f"""Localize an element on the GUI image according to the provided target and output a click position.
     * You must output a valid JSON following the format: {schema_str}
     Your target is:"""

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{prompt}\n{task}"},
            ],
        },
    ]

def get_actio_prompt(task, image):
    system_prompt = (
        "You are a GUI agent. You are given a task and a screenshot of the screen. "
        "You need to perform a series of pyautogui actions to complete the task."
    )
    instruction_text = (
        "Please perform the following task by providing the action and the coordinates in the format of <action>(x, y): "
        + task
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction_text},
                {"type": "image", "image": image},
            ],
        },
    ]

def parse_click_response(text: str) -> List[Dict]:
    actions = []
    text = text.strip()
    
    # Generic Point parsing (ActIO uses similar click(x,y) format often)
    # Looking for Click(x, y), left_click(x, y), etc.
    matches_click = re.findall(r"(?:click|left_click|right_click|double_click)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text, re.IGNORECASE)
    for m in matches_click:
        actions.append({"type": "click", "x": int(m[0]), "y": int(m[1]), "text": "", "norm": False})

    matches_point = re.findall(r"point=\[\s*(\d+)\s*,\s*(\d+)\s*\]", text, re.IGNORECASE)
    for m in matches_point:
        actions.append({"type": "click", "x": int(m[0]), "y": int(m[1]), "text": "", "norm": False})

    matches_box = re.findall(r"start_box=['\"]?\(\s*(\d+)\s*,\s*(\d+)\s*\)['\"]?", text, re.IGNORECASE)
    for m in matches_box:
        actions.append({"type": "click", "x": int(m[0]), "y": int(m[1]), "text": "", "norm": False})
    
    # Fallback tuple
    if not actions:
        matches_tuple = re.findall(r"(?:^|\s)\(\s*(\d+)\s*,\s*(\d+)\s*\)(?:$|\s|,)", text)
        for m in matches_tuple:
            actions.append({"type": "click", "x": int(m[0]), "y": int(m[1]), "text": "", "norm": False})

    return actions

def parse_fara_response(response: str) -> List[Dict]:
    actions = []
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match.strip())
            args = data.get("arguments", {})
            coords = args.get("coordinate", [])
            action_type = args.get("action", "unknown")
            text_content = args.get("text", "")
            if coords and len(coords) == 2:
                actions.append({
                    "type": action_type, "x": float(coords[0]), "y": float(coords[1]), "text": text_content, "norm": False
                })
        except Exception as e:
            print(f"Error parsing Fara JSON: {e}")
            pass
    return actions

def parse_holo2_response(response: str) -> List[Dict]:
    actions = []
    try:
        data = json.loads(response.strip())
        if 'x' in data and 'y' in data:
            actions.append({"type": "click", "x": int(data['x']), "y": int(data['y']), "text": "*", "norm": True})
            return actions
    except:
        pass

    match = re.search(r"\{\s*['\"]x['\"]\s*:\s*(\d+)\s*,\s*['\"]y['\"]\s*:\s*(\d+)\s*\}", response)
    if match:
        actions.append({
            "type": "click", 
            "x": int(match.group(1)), 
            "y": int(match.group(2)), 
            "text": "Holo2", 
            "norm": True 
        })
        return actions
    return actions

def parse_actio_response(response: str) -> List[Dict]:
    # Expected format: <action>(x, y) e.g., click(551, 355)
    # It might also just output "click(551, 355)" or "left_click(551, 355)"
    actions = []
    # General regex for name(x, y)
    matches = re.findall(r"([a-zA-Z_]+)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", response)
    for action_name, x, y in matches:
        actions.append({
            "type": action_name,
            "x": int(x),
            "y": int(y),
            "text": "",
            "norm": False # ActIO usually outputs absolute coordinates relative to input image
        })
    return actions

def create_localized_image(original_image: Image.Image, actions: list[dict]) -> Optional[Image.Image]:
    if not actions: return None
    img_copy = original_image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    try:
        font = ImageFont.load_default(size=18)
    except IOError:
        font = ImageFont.load_default()
    
    for act in actions:
        x = act['x']
        y = act['y']
        
        pixel_x, pixel_y = int(x), int(y)
            
        color = 'red' if 'click' in act['type'].lower() else 'blue'
        
        # Draw Crosshair
        line_len = 15
        width = 4
        # Horizontal
        draw.line((pixel_x - line_len, pixel_y, pixel_x + line_len, pixel_y), fill=color, width=width)
        # Vertical
        draw.line((pixel_x, pixel_y - line_len, pixel_x, pixel_y + line_len), fill=color, width=width)
        
        # Outer Circle
        r = 20
        draw.ellipse([pixel_x - r, pixel_y - r, pixel_x + r, pixel_y + r], outline=color, width=3)
        
        label = f"{act['type']}"
        if act.get('text'): label += f": \"{act['text']}\""
        
        text_pos = (pixel_x + 25, pixel_y - 15)
        
        # Label with background
        try:
            bbox = draw.textbbox(text_pos, label, font=font)
            padded_bbox = (bbox[0]-4, bbox[1]-2, bbox[2]+4, bbox[3]+2)
            draw.rectangle(padded_bbox, fill="yellow", outline=color)
            draw.text(text_pos, label, fill="black", font=font)
        except Exception as e:
            draw.text(text_pos, label, fill="white")

    return img_copy

@spaces.GPU
def process_screenshot(input_numpy_image: np.ndarray, task: str, model_choice: str):
    if input_numpy_image is None: return "⚠️ Please upload an image.", None
    if not task.strip(): return "⚠️ Please provide a task instruction.", None

    input_pil_image = array_to_image(input_numpy_image)
    orig_w, orig_h = input_pil_image.size
    actions = []
    raw_response = ""

    if model_choice == "Fara-7B":
        if model_v is None: return "Error: Fara model failed to load.", None
        print("Using Fara Pipeline...")
        
        messages = get_fara_prompt(task, input_pil_image)
        text_prompt = processor_v.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor_v(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(device)
        
        with torch.no_grad():
            generated_ids = model_v.generate(**inputs, max_new_tokens=512)
            
        generated_ids = trim_generated(generated_ids, inputs)
        raw_response = processor_v.batch_decode(generated_ids, skip_special_tokens=True)[0]
        actions = parse_fara_response(raw_response)

    elif model_choice == "Holo2-4B":
        if model_h is None: return "Error: Holo2 model failed to load.", None
        print("Using Holo2-4B Pipeline...")
        
        model, processor = model_h, processor_h
        ip_params = get_image_proc_params(processor)
        
        resized_h, resized_w = smart_resize(
            input_pil_image.height, input_pil_image.width,
            factor=ip_params["patch_size"] * ip_params["merge_size"],
            min_pixels=ip_params["min_pixels"], 
            max_pixels=ip_params["max_pixels"]
        )
        proc_image = input_pil_image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        
        messages = get_holo2_prompt(task, proc_image)
        text_prompt = apply_chat_template_compat(processor, messages, thinking=False)
        
        inputs = processor(text=[text_prompt], images=[proc_image], padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            
        generated_ids = trim_generated(generated_ids, inputs)
        raw_response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        actions = parse_holo2_response(raw_response)
        
        # Scale Holo2 coordinates (Normalized 0-1000 -> Original Pixel)
        for a in actions:
            if a.get('norm', False):
                a['x'] = (a['x'] / 1000.0) * orig_w
                a['y'] = (a['y'] / 1000.0) * orig_h

    elif model_choice == "UI-TARS-1.5-7B":
        if model_x is None: return "Error: UI-TARS model failed to load.", None
        print("Using UI-TARS Pipeline...")
        
        model, processor = model_x, processor_x
        ip_params = get_image_proc_params(processor)
        
        resized_h, resized_w = smart_resize(
            input_pil_image.height, input_pil_image.width,
            factor=ip_params["patch_size"] * ip_params["merge_size"],
            min_pixels=ip_params["min_pixels"], 
            max_pixels=ip_params["max_pixels"]
        )
        proc_image = input_pil_image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        
        messages = get_localization_prompt(task, proc_image)
        text_prompt = apply_chat_template_compat(processor, messages)
        
        inputs = processor(text=[text_prompt], images=[proc_image], padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            
        generated_ids = trim_generated(generated_ids, inputs)
        raw_response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        actions = parse_click_response(raw_response)
        
        # Scale UI-TARS coordinates (Resized Pixel -> Original Pixel)
        if resized_w > 0 and resized_h > 0:
            scale_x = orig_w / resized_w
            scale_y = orig_h / resized_h
            for a in actions:
                a['x'] = int(a['x'] * scale_x)
                a['y'] = int(a['y'] * scale_y)

    elif model_choice == "ActIO-UI-7B":
        if model_act is None: return "Error: ActIO model failed to load.", None
        print("Using ActIO-UI Pipeline...")

        model, processor = model_act, processor_act
        
        # ActIO generally uses Qwen2-VL like processing
        # We need to construct the prompt with text and image
        messages = get_actio_prompt(task, input_pil_image)
        
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # ActIO typically works with standard RGB images
        inputs = processor(
            text=[text_prompt], 
            images=[input_pil_image], 
            padding=True, 
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024, # ActIO allows verbose output sometimes
                do_sample=False,
            )

        generated_ids = trim_generated(generated_ids, inputs)
        raw_response = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        actions = parse_actio_response(raw_response)
        
        # ActIO usually outputs absolute coordinates based on the input image resolution provided to the processor.
        # Since we passed the original PIL image (unless resized internally by processor to something widely different),
        # these coords are usually correct. If ActIO resizes internally and outputs coords relative to resize, 
        # we might need scaling, but standard usage implies absolute.
        pass

    else:
        return f"Error: Unknown model '{model_choice}'", None

    print(f"Raw Output: {raw_response}")
    print(f"Parsed Actions: {actions}")

    output_image = input_pil_image
    if actions:
        vis = create_localized_image(input_pil_image, actions)
        if vis: output_image = vis
            
    return raw_response, output_image

css="""
#col-container {
    margin: 0 auto;
    max-width: 960px;
}
#main-title h1 {font-size: 2.1em !important;}
"""
with gr.Blocks() as demo:
    gr.Markdown("# **CUA GUI Operator 🖥️**", elem_id="main-title")
    gr.Markdown("Perform Computer Use Agent tasks with the models: [Fara-7B](https://huggingface.co/microsoft/Fara-7B), [UI-TARS-1.5-7B](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B), [Holo2-4B](https://huggingface.co/Hcompany/Holo2-4B), and [ActIO-UI-7B](https://huggingface.co/Uniphore/actio-ui-7b-rlvr).")

    with gr.Row():
        with gr.Column(scale=2):
            input_image = gr.Image(label="Upload UI Image", type="numpy", height=500)
            
            with gr.Row():
                model_choice = gr.Radio(
                    choices=["Fara-7B", "UI-TARS-1.5-7B", "Holo2-4B", "ActIO-UI-7B"],
                    label="Select Model",
                    value="Fara-7B",
                    interactive=True
                )
            
            task_input = gr.Textbox(
                label="Task Instruction",
                placeholder="e.g. Click on the search bar",
                lines=2
            )
            submit_btn = gr.Button("Call CUA Agent", variant="primary")

        with gr.Column(scale=3):
            output_image = gr.Image(label="Visualized Action Points", elem_id="out_img", height=500)
            output_text = gr.Textbox(label="Agent Model Response", lines=10)

    submit_btn.click(
        fn=process_screenshot,
        inputs=[input_image, task_input, model_choice],
        outputs=[output_text, output_image]
    )
    
    gr.Examples(
        examples=[
            ["examples/1.png", "Click on the Fara-7B model.", "Fara-7B"],
            ["examples/2.png", "Click on the VLMs Collection", "UI-TARS-1.5-7B"],
            ["examples/3.png", "Click on the 'SAM3'.", "Holo2-4B"],
            ["examples/1.png", "Click on the Fara-7B model.", "ActIO-UI-7B"],
        ],
        inputs=[input_image, task_input, model_choice],
        label="Quick Examples"
    )

if __name__ == "__main__":
    demo.queue(max_size=50).launch(theme=orange_red_theme, css=css, mcp_server=True, ssr_mode=False, show_error=True)