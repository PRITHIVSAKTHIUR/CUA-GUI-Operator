import os
import re
import gc
import json
import time
import base64
from io import BytesIO
from threading import Thread
from typing import List, Dict, Any, Optional

import gradio as gr
import numpy as np
import torch
import spaces
from PIL import Image, ImageDraw, ImageFont

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from qwen_vl_utils import process_vision_info

ACCENT = "#FFFF00"
MAX_INPUT_TEXT_LENGTH = int(os.getenv("MAX_INPUT_TEXT_LENGTH", "2048"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Running on device:", device)
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("🔄 Loading Fara-7B...")
MODEL_ID_V = "microsoft/Fara-7B"
try:
    processor_v = AutoProcessor.from_pretrained(MODEL_ID_V, trust_remote_code=True)
    model_v = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID_V,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
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
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
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
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device).eval()
except Exception as e:
    print(f"Failed to load Holo2: {e}")
    model_h = None
    processor_h = None

print("🔄 Loading ActIO-UI-7B...")
MODEL_ID_ACT = "Uniphore/actio-ui-7b-rlvr"
try:
    processor_act = AutoProcessor.from_pretrained(MODEL_ID_ACT, trust_remote_code=True)
    model_act = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID_ACT,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=None
    ).to(device).eval()
except Exception as e:
    print(f"Failed to load ActIO-UI: {e}")
    model_act = None
    processor_act = None

print("✅ Models loading sequence complete.")

MODEL_MAP = {
    "Fara-7B": (processor_v, model_v),
    "UI-TARS-1.5-7B": (processor_x, model_x),
    "Holo2-4B": (processor_h, model_h),
    "ActIO-UI-7B": (processor_act, model_act),
}
MODEL_CHOICES = list(MODEL_MAP.keys())

image_examples = [
    {"query": "Click on the Fara-7B model.", "image": "examples/1.png", "model": "Fara-7B"},
    {"query": "Click on the VLMs Collection", "image": "examples/2.png", "model": "UI-TARS-1.5-7B"},
    {"query": "Click on the 'SAM3'.", "image": "examples/3.png", "model": "Holo2-4B"},
    {"query": "Click on the Fara-7B model.", "image": "examples/1.png", "model": "ActIO-UI-7B"},
]

def pil_to_data_url(img: Image.Image, fmt="PNG"):
    buf = BytesIO()
    img.save(buf, format=fmt)
    data = base64.b64encode(buf.getvalue()).decode()
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{data}"

def file_to_data_url(path):
    if not os.path.exists(path):
        return ""
    ext = path.rsplit(".", 1)[-1].lower()
    mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"

def make_thumb_b64(path, max_dim=240):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_dim, max_dim))
        return pil_to_data_url(img, "JPEG")
    except Exception as e:
        print("Thumbnail error:", e)
        return ""

def b64_to_pil(b64_str):
    if not b64_str:
        return None
    try:
        if b64_str.startswith("data:"):
            _, data = b64_str.split(",", 1)
        else:
            data = b64_str
        image_data = base64.b64decode(data)
        return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception:
        return None

def build_example_cards_html():
    cards = ""
    for i, ex in enumerate(image_examples):
        thumb = make_thumb_b64(ex["image"])
        prompt_short = ex["query"][:72] + ("..." if len(ex["query"]) > 72 else "")
        cards += f"""
        <div class="example-card" data-idx="{i}">
            <div class="example-thumb-wrap">
                {"<img src='" + thumb + "' alt=''>" if thumb else "<div class='example-thumb-placeholder'>Preview</div>"}
            </div>
            <div class="example-meta-row">
                <span class="example-badge">{ex["model"]}</span>
            </div>
            <div class="example-prompt-text">{prompt_short}</div>
        </div>
        """
    return cards

EXAMPLE_CARDS_HTML = build_example_cards_html()

def load_example_data(idx_str):
    try:
        idx = int(str(idx_str).strip())
    except Exception:
        return gr.update(value=json.dumps({"status": "error", "message": "Invalid example index"}))

    if idx < 0 or idx >= len(image_examples):
        return gr.update(value=json.dumps({"status": "error", "message": "Example index out of range"}))

    ex = image_examples[idx]
    img_b64 = file_to_data_url(ex["image"])
    if not img_b64:
        return gr.update(value=json.dumps({"status": "error", "message": "Could not load example image"}))

    return gr.update(value=json.dumps({
        "status": "ok",
        "query": ex["query"],
        "image": img_b64,
        "model": ex["model"],
        "name": os.path.basename(ex["image"]),
    }))

def get_image_proc_params(processor) -> Dict[str, int]:
    ip = getattr(processor, "image_processor", None)
    default_min = 256 * 256
    default_max = 1280 * 1280
    patch_size = getattr(ip, "patch_size", 14)
    merge_size = getattr(ip, "merge_size", 2)
    min_pixels = getattr(ip, "min_pixels", default_min)
    max_pixels = getattr(ip, "max_pixels", default_max)

    size_config = getattr(ip, "size", {})
    if isinstance(size_config, dict):
        if "shortest_edge" in size_config:
            min_pixels = size_config["shortest_edge"]
        if "longest_edge" in size_config:
            max_pixels = size_config["longest_edge"]

    if min_pixels is None:
        min_pixels = default_min
    if max_pixels is None:
        max_pixels = default_max

    return {
        "patch_size": patch_size,
        "merge_size": merge_size,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
    }

def apply_chat_template_compat(processor, messages: List[Dict[str, Any]], thinking: bool = True) -> str:
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, thinking=thinking)
        except TypeError:
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
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"{guidelines}\n{task}"}
        ],
    }]

def get_holo2_prompt(task, image):
    schema_str = '{"properties": {"x": {"description": "The x coordinate, normalized between 0 and 1000.", "ge": 0, "le": 1000, "title": "X", "type": "integer"}, "y": {"description": "The y coordinate, normalized between 0 and 1000.", "ge": 0, "le": 1000, "title": "Y", "type": "integer"}}, "required": ["x", "y"], "title": "ClickCoordinates", "type": "object"}'
    prompt = f"""Localize an element on the GUI image according to the provided target and output a click position.
 * You must output a valid JSON following the format: {schema_str}
 Your target is:"""
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"{prompt}\n{task}"},
        ],
    }]

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

    matches_click = re.findall(r"(?:click|left_click|right_click|double_click)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text, re.IGNORECASE)
    for m in matches_click:
        actions.append({"type": "click", "x": int(m[0]), "y": int(m[1]), "text": "", "norm": False})

    matches_point = re.findall(r"point=\[\s*(\d+)\s*,\s*(\d+)\s*\]", text, re.IGNORECASE)
    for m in matches_point:
        actions.append({"type": "click", "x": int(m[0]), "y": int(m[1]), "text": "", "norm": False})

    matches_box = re.findall(r"start_box=['\"]?\(\s*(\d+)\s*,\s*(\d+)\s*\)['\"]?", text, re.IGNORECASE)
    for m in matches_box:
        actions.append({"type": "click", "x": int(m[0]), "y": int(m[1]), "text": "", "norm": False})

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
    return actions

def parse_holo2_response(response: str) -> List[Dict]:
    actions = []
    try:
        data = json.loads(response.strip())
        if "x" in data and "y" in data:
            actions.append({"type": "click", "x": int(data["x"]), "y": int(data["y"]), "text": "*", "norm": True})
            return actions
    except Exception:
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

def parse_actio_response(response: str) -> List[Dict]:
    actions = []
    matches = re.findall(r"([a-zA-Z_]+)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", response)
    for action_name, x, y in matches:
        actions.append({
            "type": action_name,
            "x": int(x),
            "y": int(y),
            "text": "",
            "norm": False
        })
    return actions

def create_localized_image(original_image: Image.Image, actions: List[Dict]) -> Optional[Image.Image]:
    if not actions:
        return original_image

    img_copy = original_image.copy()
    draw = ImageDraw.Draw(img_copy)

    try:
        font = ImageFont.load_default(size=18)
    except Exception:
        font = ImageFont.load_default()

    for act in actions:
        x = int(act["x"])
        y = int(act["y"])
        color = "#ff3333" if "click" in act["type"].lower() else "#3b82f6"

        line_len = 15
        width = 4

        draw.line((x - line_len, y, x + line_len, y), fill=color, width=width)
        draw.line((x, y - line_len, x, y + line_len), fill=color, width=width)

        r = 20
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)

        label = f"{act['type']}"
        if act.get("text"):
            label += f': "{act["text"]}"'

        text_pos = (x + 25, y - 15)
        try:
            bbox = draw.textbbox(text_pos, label, font=font)
            padded_bbox = (bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2)
            draw.rectangle(padded_bbox, fill="yellow", outline=color)
            draw.text(text_pos, label, fill="black", font=font)
        except Exception:
            draw.text(text_pos, label, fill="white", font=font)

    return img_copy

def calc_timeout_process(*args, **kwargs):
    gpu_timeout = kwargs.get("gpu_timeout", None)
    if gpu_timeout is None and args:
        gpu_timeout = args[-1]
    try:
        return int(gpu_timeout)
    except Exception:
        return 60

@spaces.GPU(duration=calc_timeout_process)
def process_screenshot_stream(model_choice: str, task: str, image: Image.Image, gpu_timeout: int = 60):
    try:
        if image is None:
            yield json.dumps({"status": "error", "text": "[ERROR] Please upload an image.", "annotated": ""})
            return
        if not task or not task.strip():
            yield json.dumps({"status": "error", "text": "[ERROR] Please provide a task instruction.", "annotated": ""})
            return
        if len(str(task)) > MAX_INPUT_TEXT_LENGTH * 8:
            yield json.dumps({"status": "error", "text": "[ERROR] Task instruction is too long.", "annotated": ""})
            return
        if model_choice not in MODEL_MAP:
            yield json.dumps({"status": "error", "text": "[ERROR] Invalid model selected.", "annotated": ""})
            return

        input_pil_image = image.convert("RGB")
        orig_w, orig_h = input_pil_image.size
        raw_response = ""
        actions = []

        if model_choice == "Fara-7B":
            if model_v is None:
                yield json.dumps({"status": "error", "text": "[ERROR] Fara model failed to load.", "annotated": ""})
                return

            messages = get_fara_prompt(task, input_pil_image)
            text_prompt = processor_v.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor_v(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                generated_ids = model_v.generate(**inputs, max_new_tokens=512)

            generated_ids = trim_generated(generated_ids, inputs)
            raw_response = processor_v.batch_decode(generated_ids, skip_special_tokens=True)[0]
            actions = parse_fara_response(raw_response)

        elif model_choice == "Holo2-4B":
            if model_h is None:
                yield json.dumps({"status": "error", "text": "[ERROR] Holo2 model failed to load.", "annotated": ""})
                return

            ip_params = get_image_proc_params(processor_h)
            resized_h, resized_w = smart_resize(
                input_pil_image.height,
                input_pil_image.width,
                factor=ip_params["patch_size"] * ip_params["merge_size"],
                min_pixels=ip_params["min_pixels"],
                max_pixels=ip_params["max_pixels"]
            )
            proc_image = input_pil_image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)

            messages = get_holo2_prompt(task, proc_image)
            text_prompt = apply_chat_template_compat(processor_h, messages, thinking=False)

            inputs = processor_h(text=[text_prompt], images=[proc_image], padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model_h.generate(**inputs, max_new_tokens=128)

            generated_ids = trim_generated(generated_ids, inputs)
            raw_response = processor_h.batch_decode(generated_ids, skip_special_tokens=True)[0]
            actions = parse_holo2_response(raw_response)

            for a in actions:
                if a.get("norm", False):
                    a["x"] = (a["x"] / 1000.0) * orig_w
                    a["y"] = (a["y"] / 1000.0) * orig_h

        elif model_choice == "UI-TARS-1.5-7B":
            if model_x is None:
                yield json.dumps({"status": "error", "text": "[ERROR] UI-TARS model failed to load.", "annotated": ""})
                return

            ip_params = get_image_proc_params(processor_x)
            resized_h, resized_w = smart_resize(
                input_pil_image.height,
                input_pil_image.width,
                factor=ip_params["patch_size"] * ip_params["merge_size"],
                min_pixels=ip_params["min_pixels"],
                max_pixels=ip_params["max_pixels"]
            )
            proc_image = input_pil_image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)

            messages = get_localization_prompt(task, proc_image)
            text_prompt = apply_chat_template_compat(processor_x, messages)

            inputs = processor_x(text=[text_prompt], images=[proc_image], padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model_x.generate(**inputs, max_new_tokens=128)

            generated_ids = trim_generated(generated_ids, inputs)
            raw_response = processor_x.batch_decode(generated_ids, skip_special_tokens=True)[0]
            actions = parse_click_response(raw_response)

            if resized_w > 0 and resized_h > 0:
                scale_x = orig_w / resized_w
                scale_y = orig_h / resized_h
                for a in actions:
                    a["x"] = int(a["x"] * scale_x)
                    a["y"] = int(a["y"] * scale_y)

        elif model_choice == "ActIO-UI-7B":
            if model_act is None:
                yield json.dumps({"status": "error", "text": "[ERROR] ActIO model failed to load.", "annotated": ""})
                return

            messages = get_actio_prompt(task, input_pil_image)
            text_prompt = processor_act.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = processor_act(
                text=[text_prompt],
                images=[input_pil_image],
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model_act.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                )

            generated_ids = trim_generated(generated_ids, inputs)
            raw_response = processor_act.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            actions = parse_actio_response(raw_response)

        annotated_image = create_localized_image(input_pil_image, actions)
        annotated_b64 = pil_to_data_url(annotated_image, "JPEG") if annotated_image else pil_to_data_url(input_pil_image, "JPEG")

        yield json.dumps({
            "status": "done",
            "text": raw_response,
            "annotated": annotated_b64
        })

    except Exception as e:
        yield json.dumps({"status": "error", "text": f"[ERROR] {str(e)}", "annotated": ""})
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_cua(model_name, text, image_b64, gpu_timeout_v):
    try:
        image = b64_to_pil(image_b64)
        yield from process_screenshot_stream(
            model_choice=model_name,
            task=text,
            image=image,
            gpu_timeout=gpu_timeout_v,
        )
    except Exception as e:
        yield json.dumps({"status": "error", "text": f"[ERROR] {str(e)}", "annotated": ""})

def noop():
    return None

CUBE_SVG = """
<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
  <path fill="white" d="M12 2 4 6v12l8 4 8-4V6l-8-4Zm0 2.2 5.6 2.8L12 9.8 6.4 7 12 4.2Zm-6 4.5 5 2.5v8.6l-5-2.5V8.7Zm7 11.1v-8.6l5-2.5v8.6l-5 2.5Z"/>
</svg>
"""

UPLOAD_PREVIEW_SVG = f"""
<svg viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="8" y="14" width="64" height="52" rx="6" fill="none" stroke="{ACCENT}" stroke-width="2" stroke-dasharray="4 3"/>
    <polygon points="12,62 30,40 42,50 54,34 68,62" fill="rgba(255,255,0,0.14)" stroke="{ACCENT}" stroke-width="1.5"/>
    <circle cx="28" cy="30" r="6" fill="rgba(255,255,0,0.2)" stroke="{ACCENT}" stroke-width="1.5"/>
</svg>
"""

ANNOTATION_PLACEHOLDER_SVG = f"""
<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg" fill="none">
  <path d="M60 16 24 34v52l36 18 36-18V34L60 16Z" stroke="{ACCENT}" stroke-width="3"/>
  <path d="M24 34 60 52l36-18M60 52v52" stroke="{ACCENT}" stroke-width="2.5"/>
</svg>
"""

COPY_SVG = f"""<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="{ACCENT}" d="M16 1H4C2.9 1 2 1.9 2 3v12h2V3h12V1zm3 4H8C6.9 5 6 5.9 6 7v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>"""
SAVE_SVG = f"""<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="{ACCENT}" d="M17 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V7l-4-4zM7 5h8v4H7V5zm12 14H5v-6h14v6z"/></svg>"""

MODEL_TABS_HTML = "".join([
    f'<button class="model-tab{" active" if m == "Fara-7B" else ""}" data-model="{m}"><span class="model-tab-label">{m}</span></button>'
    for m in MODEL_CHOICES
])


css = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{height:100%;overflow-x:hidden}}
body,.gradio-container{{
    background:#0f0f13!important;
    font-family:'Inter',system-ui,-apple-system,sans-serif!important;
    font-size:14px!important;color:#e4e4e7!important;min-height:100vh;overflow-x:hidden;
}}
.dark body,.dark .gradio-container{{background:#0f0f13!important;color:#e4e4e7!important}}
footer{{display:none!important}}
.hidden-input{{display:none!important;height:0!important;overflow:hidden!important;margin:0!important;padding:0!important}}
#gradio-run-btn,#example-load-btn{{
    position:absolute!important;left:-9999px!important;top:-9999px!important;
    width:1px!important;height:1px!important;opacity:0.01!important;
    pointer-events:none!important;overflow:hidden!important;
}}

.app-shell{{
    background:#18181b;border:1px solid #27272a;border-radius:16px;
    margin:12px auto;max-width:1440px;overflow:hidden;
    box-shadow:0 25px 50px -12px rgba(0,0,0,.6),0 0 0 1px rgba(255,255,255,.03);
}}
.app-header{{
    background:linear-gradient(135deg,#18181b,#1e1e24);border-bottom:1px solid #27272a;
    padding:14px 24px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;
}}
.app-header-left{{display:flex;align-items:center;gap:12px}}
.app-logo{{
    width:38px;height:38px;background:linear-gradient(135deg,{ACCENT},#fff06a,#fff7b2);
    border-radius:10px;display:flex;align-items:center;justify-content:center;
    box-shadow:0 4px 12px rgba(255,255,0,.30);
}}
.app-logo svg{{width:22px;height:22px;fill:#111;flex-shrink:0}}
.app-title{{
    font-size:18px;font-weight:700;background:linear-gradient(135deg,#f5f5f5,#d9d9a7);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-.3px;
}}
.app-badge{{
    font-size:11px;font-weight:600;padding:3px 10px;border-radius:20px;
    background:rgba(255,255,0,.10);color:#fff8a6;border:1px solid rgba(255,255,0,.24);letter-spacing:.3px;
}}
.app-badge.fast{{background:rgba(255,255,0,.08);color:#fff39a;border:1px solid rgba(255,255,0,.20)}}

.model-tabs-bar{{
    background:#18181b;border-bottom:1px solid #27272a;padding:10px 16px;
    display:flex;gap:8px;align-items:center;flex-wrap:wrap;
}}
.model-tab{{
    display:inline-flex;align-items:center;justify-content:center;gap:6px;
    min-width:32px;height:34px;background:transparent;border:1px solid #27272a;
    border-radius:999px;cursor:pointer;font-size:12px;font-weight:600;padding:0 12px;
    color:#ffffff!important;transition:all .15s ease;
}}
.model-tab:hover{{background:rgba(255,255,0,.10);border-color:rgba(255,255,0,.35)}}
.model-tab.active{{background:rgba(255,255,0,.16);border-color:{ACCENT};color:#fff!important;box-shadow:0 0 0 2px rgba(255,255,0,.08)}}
.model-tab-label{{font-size:12px;color:#ffffff!important;font-weight:600}}

.app-main-row{{display:flex;gap:0;flex:1;overflow:hidden}}
.app-main-left{{flex:1;display:flex;flex-direction:column;min-width:0;border-right:1px solid #27272a}}
.app-main-right{{width:520px;display:flex;flex-direction:column;flex-shrink:0;background:#18181b}}

#image-drop-zone{{
    position:relative;background:#09090b;height:460px;min-height:460px;max-height:460px;
    overflow:hidden;
}}
#image-drop-zone.drag-over{{outline:2px solid {ACCENT};outline-offset:-2px;background:rgba(255,255,0,.04)}}
.upload-prompt-modern{{
    position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
    padding:20px;z-index:20;overflow:hidden;
}}
.upload-click-area{{
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    cursor:pointer;padding:28px 36px;max-width:92%;max-height:92%;
    border:2px dashed #3f3f46;border-radius:16px;
    background:rgba(255,255,0,.03);transition:all .2s ease;gap:8px;text-align:center;
    overflow:hidden;
}}
.upload-click-area:hover{{background:rgba(255,255,0,.08);border-color:{ACCENT};transform:scale(1.02)}}
.upload-click-area:active{{background:rgba(255,255,0,.12);transform:scale(.99)}}
.upload-click-area svg{{width:86px;height:86px;max-width:100%;flex-shrink:0}}
.upload-main-text{{color:#a1a1aa;font-size:14px;font-weight:600;margin-top:4px}}
.upload-sub-text{{color:#71717a;font-size:12px}}

.single-preview-wrap{{
    width:100%;height:100%;display:none;align-items:center;justify-content:center;padding:16px;
    overflow:hidden;
}}
.single-preview-card{{
    width:100%;height:100%;max-width:100%;max-height:100%;border-radius:14px;
    overflow:hidden;border:1px solid #27272a;background:#111114;
    display:flex;align-items:center;justify-content:center;position:relative;
}}
.single-preview-card img{{
    width:100%;height:100%;max-width:100%;max-height:100%;
    object-fit:contain;display:block;
}}
.preview-overlay-actions{{
    position:absolute;top:12px;right:12px;display:flex;gap:8px;z-index:5;
}}
.preview-action-btn{{
    display:inline-flex;align-items:center;justify-content:center;
    min-width:34px;height:34px;padding:0 12px;background:rgba(0,0,0,.65);
    border:1px solid rgba(255,255,255,.14);border-radius:10px;cursor:pointer;
    color:#fff!important;font-size:12px;font-weight:600;transition:all .15s ease;
}}
.preview-action-btn:hover{{background:{ACCENT};border-color:{ACCENT};color:#121200!important}}

.hint-bar{{
    background:rgba(255,255,0,.05);border-top:1px solid #27272a;border-bottom:1px solid #27272a;
    padding:10px 20px;font-size:13px;color:#a1a1aa;line-height:1.7;
}}
.hint-bar b{{color:#fff6a0;font-weight:600}}
.hint-bar kbd{{
    display:inline-block;padding:1px 6px;background:#27272a;border:1px solid #3f3f46;
    border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#a1a1aa;
}}

.examples-section{{border-top:1px solid #27272a;padding:12px 16px}}
.examples-title{{
    font-size:12px;font-weight:600;color:#71717a;text-transform:uppercase;
    letter-spacing:.8px;margin-bottom:10px;
}}
.examples-scroll{{display:flex;gap:10px;overflow-x:auto;padding-bottom:8px}}
.examples-scroll::-webkit-scrollbar{{height:6px}}
.examples-scroll::-webkit-scrollbar-track{{background:#09090b;border-radius:3px}}
.examples-scroll::-webkit-scrollbar-thumb{{background:#27272a;border-radius:3px}}
.examples-scroll::-webkit-scrollbar-thumb:hover{{background:#3f3f46}}
.example-card{{
    flex-shrink:0;width:220px;background:#09090b;border:1px solid #27272a;
    border-radius:10px;overflow:hidden;cursor:pointer;transition:all .2s ease;
}}
.example-card:hover{{border-color:{ACCENT};transform:translateY(-2px);box-shadow:0 4px 12px rgba(255,255,0,.14)}}
.example-card.loading{{opacity:.5;pointer-events:none}}
.example-thumb-wrap{{height:120px;overflow:hidden;background:#18181b}}
.example-thumb-wrap img{{width:100%;height:100%;object-fit:cover}}
.example-thumb-placeholder{{
    width:100%;height:100%;display:flex;align-items:center;justify-content:center;
    background:#18181b;color:#3f3f46;font-size:11px;
}}
.example-meta-row{{padding:6px 10px;display:flex;align-items:center;gap:6px}}
.example-badge{{
    display:inline-flex;padding:2px 7px;background:rgba(255,255,0,.12);border-radius:4px;
    font-size:10px;font-weight:600;color:#fff6a0;font-family:'JetBrains Mono',monospace;white-space:nowrap;
}}
.example-prompt-text{{
    padding:0 10px 8px;font-size:11px;color:#a1a1aa;line-height:1.4;
    display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;
}}

.panel-card{{border-bottom:1px solid #27272a}}
.panel-card-title{{
    padding:12px 20px;font-size:12px;font-weight:600;color:#71717a;
    text-transform:uppercase;letter-spacing:.8px;border-bottom:1px solid rgba(39,39,42,.6);
}}
.panel-card-body{{padding:16px 20px;display:flex;flex-direction:column;gap:8px}}
.modern-label{{font-size:13px;font-weight:500;color:#a1a1aa;margin-bottom:4px;display:block}}
.modern-textarea{{
    width:100%;background:#09090b;border:1px solid #27272a;border-radius:8px;
    padding:10px 14px;font-family:'Inter',sans-serif;font-size:14px;color:#e4e4e7;
    resize:none;outline:none;min-height:100px;transition:border-color .2s;
}}
.modern-textarea:focus{{border-color:{ACCENT};box-shadow:0 0 0 3px rgba(255,255,0,.14)}}
.modern-textarea::placeholder{{color:#3f3f46}}
.modern-textarea.error-flash{{
    border-color:#ef4444!important;box-shadow:0 0 0 3px rgba(239,68,68,.2)!important;animation:shake .4s ease;
}}
@keyframes shake{{0%,100%{{transform:translateX(0)}}20%,60%{{transform:translateX(-4px)}}40%,80%{{transform:translateX(4px)}}}}

.toast-notification{{
    position:fixed;top:24px;left:50%;transform:translateX(-50%) translateY(-120%);
    z-index:9999;padding:10px 24px;border-radius:10px;font-family:'Inter',sans-serif;
    font-size:14px;font-weight:600;display:flex;align-items:center;gap:8px;
    box-shadow:0 8px 24px rgba(0,0,0,.5);
    transition:transform .35s cubic-bezier(.34,1.56,.64,1),opacity .35s ease;opacity:0;pointer-events:none;
}}
.toast-notification.visible{{transform:translateX(-50%) translateY(0);opacity:1;pointer-events:auto}}
.toast-notification.error{{background:linear-gradient(135deg,#dc2626,#b91c1c);color:#fff;border:1px solid rgba(255,255,255,.15)}}
.toast-notification.warning{{background:linear-gradient(135deg,#b7b700,#8f8f00);color:#fff;border:1px solid rgba(255,255,255,.15)}}
.toast-notification.info{{background:linear-gradient(135deg,#d4d400,{ACCENT});color:#111;border:1px solid rgba(255,255,255,.15)}}
.toast-notification .toast-icon{{font-size:16px;line-height:1}}
.toast-notification .toast-text{{line-height:1.3}}

.btn-run{{
    display:flex;align-items:center;justify-content:center;gap:8px;width:100%;
    background:linear-gradient(135deg,{ACCENT},#d8d800);border:none;border-radius:10px;
    padding:12px 24px;cursor:pointer;font-size:15px;font-weight:700;font-family:'Inter',sans-serif;
    color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;
    transition:all .2s ease;letter-spacing:-.2px;
    box-shadow:0 4px 16px rgba(255,255,0,.25),inset 0 1px 0 rgba(255,255,255,.18);
}}
.btn-run:hover{{
    background:linear-gradient(135deg,#ffff7a,{ACCENT});transform:translateY(-1px);
    box-shadow:0 6px 24px rgba(255,255,0,.35),inset 0 1px 0 rgba(255,255,255,.22);
}}
.btn-run:active{{transform:translateY(0);box-shadow:0 2px 8px rgba(255,255,0,.25)}}

.annot-frame{{border-bottom:1px solid #27272a;display:flex;flex-direction:column;position:relative}}
.annot-title{{
    padding:10px 20px;font-size:13px;font-weight:700;text-transform:uppercase;
    letter-spacing:.8px;border-bottom:1px solid rgba(39,39,42,.6);color:#fff
}}
.annot-body{{
    background:#09090b;height:340px;display:flex;align-items:center;justify-content:center;
    padding:12px;position:relative;overflow:hidden;
}}
.annot-body img{{
    max-width:100%;max-height:100%;object-fit:contain;border:1px solid #27272a;
    border-radius:10px;background:#111114;display:none;position:relative;z-index:2;
}}
.annot-placeholder{{
    position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;
    gap:10px;color:#666;z-index:1;padding:16px;text-align:center;
}}
.annot-placeholder svg{{width:92px;height:92px;max-width:100%;opacity:.95}}
.annot-placeholder-title{{font-size:13px;font-weight:600;color:#fff6a0}}
.annot-placeholder-sub{{font-size:12px;color:#666;max-width:260px;line-height:1.5}}

.output-frame{{border-bottom:1px solid #27272a;display:flex;flex-direction:column;position:relative}}
.output-frame .out-title,
.output-frame .out-title *,
#output-title-label{{
    color:#ffffff!important;
    -webkit-text-fill-color:#ffffff!important;
}}
.output-frame .out-title{{
    padding:10px 20px;font-size:13px;font-weight:700;
    text-transform:uppercase;letter-spacing:.8px;border-bottom:1px solid rgba(39,39,42,.6);
    display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;
}}
.out-title-right{{display:flex;gap:8px;align-items:center}}
.out-action-btn{{
    display:inline-flex;align-items:center;justify-content:center;background:rgba(255,255,0,.10);
    border:1px solid rgba(255,255,0,.2);border-radius:6px;cursor:pointer;padding:3px 10px;
    font-size:11px;font-weight:500;color:#fff6a0!important;gap:4px;height:24px;transition:all .15s;
}}
.out-action-btn:hover{{background:rgba(255,255,0,.2);border-color:rgba(255,255,0,.35);color:#ffffff!important}}
.out-action-btn svg{{width:12px;height:12px;fill:{ACCENT}}}
.output-frame .out-body{{
    flex:1;background:#09090b;display:flex;align-items:stretch;justify-content:stretch;
    overflow:hidden;min-height:300px;position:relative;
}}
.output-scroll-wrap{{width:100%;height:100%;padding:0;overflow:hidden}}
.output-textarea{{
    width:100%;height:300px;min-height:300px;max-height:300px;background:#09090b;color:#e4e4e7;
    border:none;outline:none;padding:16px 18px;font-size:13px;line-height:1.6;
    font-family:'JetBrains Mono',monospace;overflow:auto;resize:none;white-space:pre-wrap;
}}
.output-textarea::placeholder{{color:#52525b}}
.output-textarea.error-flash{{box-shadow:inset 0 0 0 2px rgba(239,68,68,.6)}}

.modern-loader{{
    display:none;position:absolute;top:0;left:0;right:0;bottom:0;background:rgba(9,9,11,.92);
    z-index:15;flex-direction:column;align-items:center;justify-content:center;gap:16px;backdrop-filter:blur(4px);
}}
.modern-loader.active{{display:flex}}
.modern-loader .loader-spinner{{
    width:36px;height:36px;border:3px solid #27272a;border-top-color:{ACCENT};
    border-radius:50%;animation:spin .8s linear infinite;
}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
.modern-loader .loader-text{{font-size:13px;color:#a1a1aa;font-weight:500}}
.loader-bar-track{{width:200px;height:4px;background:#27272a;border-radius:2px;overflow:hidden}}
.loader-bar-fill{{
    height:100%;background:linear-gradient(90deg,{ACCENT},#ffff94,{ACCENT});
    background-size:200% 100%;animation:shimmer 1.5s ease-in-out infinite;border-radius:2px;
}}
@keyframes shimmer{{0%{{background-position:200% 0}}100%{{background-position:-200% 0}}}}

.settings-group{{border:1px solid #27272a;border-radius:10px;margin:12px 16px;padding:0;overflow:hidden}}
.settings-group-title{{
    font-size:12px;font-weight:600;color:#71717a;text-transform:uppercase;letter-spacing:.8px;
    padding:10px 16px;border-bottom:1px solid #27272a;background:rgba(24,24,27,.5);
}}
.settings-group-body{{padding:14px 16px;display:flex;flex-direction:column;gap:12px}}
.slider-row{{display:flex;align-items:center;gap:10px;min-height:28px}}
.slider-row label{{font-size:13px;font-weight:500;color:#a1a1aa;min-width:118px;flex-shrink:0}}
.slider-row input[type="range"]{{
    flex:1;-webkit-appearance:none;appearance:none;height:6px;background:#27272a;
    border-radius:3px;outline:none;min-width:0;
}}
.slider-row input[type="range"]::-webkit-slider-thumb{{
    -webkit-appearance:none;width:16px;height:16px;background:linear-gradient(135deg,{ACCENT},#d8d800);
    border-radius:50%;cursor:pointer;box-shadow:0 2px 6px rgba(255,255,0,.35);transition:transform .15s;
}}
.slider-row input[type="range"]::-webkit-slider-thumb:hover{{transform:scale(1.2)}}
.slider-row input[type="range"]::-moz-range-thumb{{
    width:16px;height:16px;background:linear-gradient(135deg,{ACCENT},#d8d800);
    border-radius:50%;cursor:pointer;border:none;box-shadow:0 2px 6px rgba(255,255,0,.35);
}}
.slider-row .slider-val{{
    min-width:58px;text-align:right;font-family:'JetBrains Mono',monospace;font-size:12px;
    font-weight:500;padding:3px 8px;background:#09090b;border:1px solid #27272a;
    border-radius:6px;color:#a1a1aa;flex-shrink:0;
}}

.app-statusbar{{
    background:#18181b;border-top:1px solid #27272a;padding:6px 20px;
    display:flex;gap:12px;height:34px;align-items:center;font-size:12px;
}}
.app-statusbar .sb-section{{
    padding:0 12px;flex:1;display:flex;align-items:center;font-family:'JetBrains Mono',monospace;
    font-size:12px;color:#52525b;overflow:hidden;white-space:nowrap;
}}
.app-statusbar .sb-section.sb-fixed{{
    flex:0 0 auto;min-width:110px;text-align:center;justify-content:center;
    padding:3px 12px;background:rgba(255,255,0,.08);border-radius:6px;color:#fff6a0;font-weight:500;
}}

.exp-note{{padding:10px 20px;font-size:12px;color:#52525b;border-top:1px solid #27272a;text-align:center}}
.exp-note a{{color:#fff6a0;text-decoration:none}}
.exp-note a:hover{{text-decoration:underline}}

::-webkit-scrollbar{{width:8px;height:8px}}
::-webkit-scrollbar-track{{background:#09090b}}
::-webkit-scrollbar-thumb{{background:#27272a;border-radius:4px}}
::-webkit-scrollbar-thumb:hover{{background:#3f3f46}}

@media(max-width:980px){{
    .app-main-row{{flex-direction:column}}
    .app-main-right{{width:100%}}
    .app-main-left{{border-right:none;border-bottom:1px solid #27272a}}
}}
"""

gallery_js = r"""
() => {
function init() {
    if (window.__cuaInitDone) return;

    const dropZone = document.getElementById('image-drop-zone');
    const uploadPrompt = document.getElementById('upload-prompt');
    const uploadClick = document.getElementById('upload-click-area');
    const fileInput = document.getElementById('custom-file-input');
    const previewWrap = document.getElementById('single-preview-wrap');
    const previewImg = document.getElementById('single-preview-img');
    const btnUpload = document.getElementById('preview-upload-btn');
    const btnClear = document.getElementById('preview-clear-btn');
    const promptInput = document.getElementById('custom-query-input');
    const runBtnEl = document.getElementById('custom-run-btn');
    const outputArea = document.getElementById('custom-output-textarea');
    const annotImg = document.getElementById('annotated-output-img');
    const annotPlaceholder = document.getElementById('annotated-output-placeholder');
    const imgStatus = document.getElementById('sb-image-status');

    if (!dropZone || !fileInput || !promptInput || !previewWrap || !previewImg) {
        setTimeout(init, 250);
        return;
    }

    window.__cuaInitDone = true;
    let imageState = null;
    let toastTimer = null;
    let examplePoller = null;
    let lastSeenExamplePayload = null;

    function showToast(message, type) {
        let toast = document.getElementById('app-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'app-toast';
            toast.className = 'toast-notification';
            toast.innerHTML = '<span class="toast-icon"></span><span class="toast-text"></span>';
            document.body.appendChild(toast);
        }
        const icon = toast.querySelector('.toast-icon');
        const text = toast.querySelector('.toast-text');
        toast.className = 'toast-notification ' + (type || 'error');
        if (type === 'warning') icon.textContent = '\u26A0';
        else if (type === 'info') icon.textContent = '\u2139';
        else icon.textContent = '\u2717';
        text.textContent = message;
        if (toastTimer) clearTimeout(toastTimer);
        void toast.offsetWidth;
        toast.classList.add('visible');
        toastTimer = setTimeout(() => toast.classList.remove('visible'), 3500);
    }

    function showLoader() {
        const l = document.getElementById('output-loader');
        if (l) l.classList.add('active');
        const sb = document.getElementById('sb-run-state');
        if (sb) sb.textContent = 'Processing...';
    }
    function hideLoader() {
        const l = document.getElementById('output-loader');
        if (l) l.classList.remove('active');
        const sb = document.getElementById('sb-run-state');
        if (sb) sb.textContent = 'Done';
    }
    function setRunErrorState() {
        const l = document.getElementById('output-loader');
        if (l) l.classList.remove('active');
        const sb = document.getElementById('sb-run-state');
        if (sb) sb.textContent = 'Error';
    }

    function flashPromptError() {
        promptInput.classList.add('error-flash');
        promptInput.focus();
        setTimeout(() => promptInput.classList.remove('error-flash'), 800);
    }

    function flashOutputError() {
        if (!outputArea) return;
        outputArea.classList.add('error-flash');
        setTimeout(() => outputArea.classList.remove('error-flash'), 800);
    }

    function getValueFromContainer(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return '';
        const el = container.querySelector('textarea, input');
        return el ? (el.value || '') : '';
    }

    function setGradioValue(containerId, value) {
        const container = document.getElementById(containerId);
        if (!container) return false;
        const el = container.querySelector('textarea, input');
        if (!el) return false;
        const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
        const ns = Object.getOwnPropertyDescriptor(proto, 'value');
        if (ns && ns.set) {
            ns.set.call(el, value);
            el.dispatchEvent(new Event('input', {bubbles:true, composed:true}));
            el.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
            return true;
        }
        return false;
    }

    function syncImageToGradio() {
        setGradioValue('hidden-image-b64', imageState ? imageState.b64 : '');
        if (imgStatus) imgStatus.textContent = imageState ? '1 image uploaded' : 'No image uploaded';
    }

    function syncPromptToGradio() {
        setGradioValue('prompt-gradio-input', promptInput.value);
    }

    function syncModelToGradio(name) {
        setGradioValue('hidden-model-name', name);
    }

    function updateAnnotationState(src) {
        if (!annotImg || !annotPlaceholder) return;
        if (src) {
            annotImg.src = src;
            annotImg.style.display = 'block';
            annotPlaceholder.style.display = 'none';
        } else {
            annotImg.src = '';
            annotImg.style.display = 'none';
            annotPlaceholder.style.display = 'flex';
        }
    }

    function setPreview(b64, name) {
        imageState = {b64, name: name || 'image'};
        previewImg.src = b64;
        previewWrap.style.display = 'flex';
        if (uploadPrompt) uploadPrompt.style.display = 'none';
        syncImageToGradio();
    }

    function clearPreview() {
        imageState = null;
        previewImg.src = '';
        previewWrap.style.display = 'none';
        if (uploadPrompt) uploadPrompt.style.display = 'flex';
        syncImageToGradio();
        updateAnnotationState('');
    }

    window.__setPreview = setPreview;
    window.__clearPreview = clearPreview;
    window.__updateAnnotationState = updateAnnotationState;
    window.__showToast = showToast;
    window.__showLoader = showLoader;
    window.__hideLoader = hideLoader;
    window.__setRunErrorState = setRunErrorState;

    function processFile(file) {
        if (!file) return;
        if (!file.type.startsWith('image/')) {
            showToast('Only image files are supported', 'error');
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => setPreview(e.target.result, file.name);
        reader.readAsDataURL(file);
    }

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files && e.target.files[0] ? e.target.files[0] : null;
        if (file) processFile(file);
        e.target.value = '';
    });

    if (uploadClick) uploadClick.addEventListener('click', () => fileInput.click());
    if (btnUpload) btnUpload.addEventListener('click', () => fileInput.click());
    if (btnClear) btnClear.addEventListener('click', clearPreview);

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files && e.dataTransfer.files.length) processFile(e.dataTransfer.files[0]);
    });

    promptInput.addEventListener('input', syncPromptToGradio);

    function activateModelTab(name) {
        document.querySelectorAll('.model-tab[data-model]').forEach(btn => {
            btn.classList.toggle('active', btn.getAttribute('data-model') === name);
        });
        syncModelToGradio(name);
    }
    window.__activateModelTab = activateModelTab;

    document.querySelectorAll('.model-tab[data-model]').forEach(btn => {
        btn.addEventListener('click', () => activateModelTab(btn.getAttribute('data-model')));
    });

    activateModelTab('Fara-7B');
    updateAnnotationState('');

    function syncSlider(customId, gradioId) {
        const slider = document.getElementById(customId);
        const valSpan = document.getElementById(customId + '-val');
        if (!slider) return;
        slider.addEventListener('input', () => {
            if (valSpan) valSpan.textContent = slider.value;
            const container = document.getElementById(gradioId);
            if (!container) return;
            container.querySelectorAll('input[type="range"],input[type="number"]').forEach(el => {
                const ns = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
                if (ns && ns.set) {
                    ns.set.call(el, slider.value);
                    el.dispatchEvent(new Event('input', {bubbles:true, composed:true}));
                    el.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
                }
            });
        });
    }

    syncSlider('custom-gpu-duration', 'gradio-gpu-duration');

    function validateBeforeRun() {
        const promptVal = promptInput.value.trim();
        if (!imageState && !promptVal) {
            showToast('Please upload an image and enter your task instruction', 'error');
            flashPromptError();
            return false;
        }
        if (!imageState) {
            showToast('Please upload an image', 'error');
            return false;
        }
        if (!promptVal) {
            showToast('Please enter your task instruction', 'warning');
            flashPromptError();
            return false;
        }
        const currentModel = (document.querySelector('.model-tab.active') || {}).dataset?.model;
        if (!currentModel) {
            showToast('Please select a model', 'error');
            return false;
        }
        return true;
    }

    window.__clickGradioRunBtn = function() {
        if (!validateBeforeRun()) return;
        syncPromptToGradio();
        syncImageToGradio();
        const active = document.querySelector('.model-tab.active');
        if (active) syncModelToGradio(active.getAttribute('data-model'));
        if (outputArea) outputArea.value = '';
        updateAnnotationState('');
        showLoader();
        setTimeout(() => {
            const gradioBtn = document.getElementById('gradio-run-btn');
            if (!gradioBtn) {
                setRunErrorState();
                if (outputArea) outputArea.value = '[ERROR] Run button not found.';
                showToast('Run button not found', 'error');
                return;
            }
            const btn = gradioBtn.querySelector('button');
            if (btn) btn.click(); else gradioBtn.click();
        }, 180);
    };

    if (runBtnEl) runBtnEl.addEventListener('click', () => window.__clickGradioRunBtn());

    const copyBtn = document.getElementById('copy-output-btn');
    if (copyBtn) {
        copyBtn.addEventListener('click', async () => {
            try {
                const text = outputArea ? outputArea.value : '';
                if (!text.trim()) {
                    showToast('No output to copy', 'warning');
                    flashOutputError();
                    return;
                }
                await navigator.clipboard.writeText(text);
                showToast('Output copied to clipboard', 'info');
            } catch(e) {
                showToast('Copy failed', 'error');
            }
        });
    }

    const saveBtn = document.getElementById('save-output-btn');
    if (saveBtn) {
        saveBtn.addEventListener('click', () => {
            const text = outputArea ? outputArea.value : '';
            if (!text.trim()) {
                showToast('No output to save', 'warning');
                flashOutputError();
                return;
            }
            const blob = new Blob([text], {type: 'text/plain;charset=utf-8'});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'cua_gui_operator_output.txt';
            document.body.appendChild(a);
            a.click();
            setTimeout(() => {
                URL.revokeObjectURL(a.href);
                document.body.removeChild(a);
            }, 200);
            showToast('Output saved', 'info');
        });
    }

    function applyExamplePayload(raw) {
        try {
            const data = JSON.parse(raw);
            if (data.status === 'ok') {
                if (data.image) setPreview(data.image, data.name || 'example.png');
                if (data.query) {
                    promptInput.value = data.query;
                    syncPromptToGradio();
                }
                if (data.model) activateModelTab(data.model);
                document.querySelectorAll('.example-card.loading').forEach(c => c.classList.remove('loading'));
                showToast('Example loaded', 'info');
            } else if (data.status === 'error') {
                document.querySelectorAll('.example-card.loading').forEach(c => c.classList.remove('loading'));
                showToast(data.message || 'Failed to load example', 'error');
            }
        } catch (e) {
            document.querySelectorAll('.example-card.loading').forEach(c => c.classList.remove('loading'));
        }
    }

    function startExamplePolling() {
        if (examplePoller) clearInterval(examplePoller);
        let attempts = 0;
        examplePoller = setInterval(() => {
            attempts += 1;
            const current = getValueFromContainer('example-result-data');
            if (current && current !== lastSeenExamplePayload) {
                lastSeenExamplePayload = current;
                clearInterval(examplePoller);
                examplePoller = null;
                applyExamplePayload(current);
                return;
            }
            if (attempts >= 100) {
                clearInterval(examplePoller);
                examplePoller = null;
                document.querySelectorAll('.example-card.loading').forEach(c => c.classList.remove('loading'));
                showToast('Example load timed out', 'error');
            }
        }, 120);
    }

    function triggerExampleLoad(idx) {
        const btnWrap = document.getElementById('example-load-btn');
        const btn = btnWrap ? (btnWrap.querySelector('button') || btnWrap) : null;
        if (!btn) return;

        let attempts = 0;
        function writeIdxAndClick() {
            attempts += 1;
            const ok1 = setGradioValue('example-idx-input', String(idx));
            setGradioValue('example-result-data', '');
            const currentVal = getValueFromContainer('example-idx-input');

            if (ok1 && currentVal === String(idx)) {
                btn.click();
                startExamplePolling();
                return;
            }

            if (attempts < 30) {
                setTimeout(writeIdxAndClick, 100);
            } else {
                document.querySelectorAll('.example-card.loading').forEach(c => c.classList.remove('loading'));
                showToast('Failed to initialize example loader', 'error');
            }
        }
        writeIdxAndClick();
    }

    document.querySelectorAll('.example-card[data-idx]').forEach(card => {
        card.addEventListener('click', () => {
            const idx = card.getAttribute('data-idx');
            if (!idx) return;
            document.querySelectorAll('.example-card.loading').forEach(c => c.classList.remove('loading'));
            card.classList.add('loading');
            showToast('Loading example...', 'info');
            triggerExampleLoad(idx);
        });
    });

    const observerTarget = document.getElementById('example-result-data');
    if (observerTarget) {
        const obs = new MutationObserver(() => {
            const current = getValueFromContainer('example-result-data');
            if (!current || current === lastSeenExamplePayload) return;
            lastSeenExamplePayload = current;
            if (examplePoller) {
                clearInterval(examplePoller);
                examplePoller = null;
            }
            applyExamplePayload(current);
        });
        obs.observe(observerTarget, {childList:true, subtree:true, characterData:true, attributes:true});
    }

    if (outputArea) outputArea.value = '';
    const sb = document.getElementById('sb-run-state');
    if (sb) sb.textContent = 'Ready';
    if (imgStatus) imgStatus.textContent = 'No image uploaded';
}
init();
}
"""

wire_outputs_js = r"""
() => {
function watchOutputs() {
    const resultContainer = document.getElementById('gradio-result');
    const outArea = document.getElementById('custom-output-textarea');

    if (!resultContainer || !outArea) { setTimeout(watchOutputs, 500); return; }

    let lastText = '';

    function syncOutput() {
        const el = resultContainer.querySelector('textarea') || resultContainer.querySelector('input');
        if (!el) return;
        const val = el.value || '';

        if (val !== lastText) {
            lastText = val;
            try {
                const data = JSON.parse(val);
                if (data.text !== undefined) {
                    outArea.value = data.text || '';
                    outArea.scrollTop = outArea.scrollHeight;
                }
                if (data.annotated && window.__updateAnnotationState) {
                    window.__updateAnnotationState(data.annotated);
                }
                if (data.status === 'error') {
                    if (window.__setRunErrorState) window.__setRunErrorState();
                    if (window.__showToast) window.__showToast('Inference failed', 'error');
                } else if (data.status === 'done') {
                    if (window.__hideLoader) window.__hideLoader();
                }
            } catch (e) {
                outArea.value = val;
                outArea.scrollTop = outArea.scrollHeight;
            }
        }
    }

    const observer = new MutationObserver(syncOutput);
    observer.observe(resultContainer, {childList:true, subtree:true, characterData:true, attributes:true});
    setInterval(syncOutput, 500);
}
watchOutputs();
}
"""

with gr.Blocks() as demo:
    hidden_image_b64 = gr.Textbox(value="", elem_id="hidden-image-b64", elem_classes="hidden-input", container=False)
    prompt = gr.Textbox(value="", elem_id="prompt-gradio-input", elem_classes="hidden-input", container=False)
    hidden_model_name = gr.Textbox(value="Fara-7B", elem_id="hidden-model-name", elem_classes="hidden-input", container=False)
    gpu_duration_state = gr.Number(value=60, elem_id="gradio-gpu-duration", elem_classes="hidden-input", container=False)

    result = gr.Textbox(value="", elem_id="gradio-result", elem_classes="hidden-input", container=False)

    example_idx = gr.Textbox(value="", elem_id="example-idx-input", elem_classes="hidden-input", container=False)
    example_result = gr.Textbox(value="", elem_id="example-result-data", elem_classes="hidden-input", container=False)
    example_load_btn = gr.Button("Load Example", elem_id="example-load-btn")

    gr.HTML(f"""
    <div class="app-shell">
        <div class="app-header">
            <div class="app-header-left">
                <div class="app-logo">{CUBE_SVG}</div>
                <span class="app-title">CUA GUI Operator</span>
                <span class="app-badge">computer use</span>
                <span class="app-badge fast">visual action grounding</span>
            </div>
        </div>

        <div class="model-tabs-bar">
            {MODEL_TABS_HTML}
        </div>

        <div class="app-main-row">
            <div class="app-main-left">
                <div id="image-drop-zone">
                    <div id="upload-prompt" class="upload-prompt-modern">
                        <div id="upload-click-area" class="upload-click-area">
                            {UPLOAD_PREVIEW_SVG}
                            <span class="upload-main-text">Click or drag a UI screenshot here</span>
                            <span class="upload-sub-text">Upload one interface screenshot for computer-use action localization, click grounding, or agent-style next-step prediction</span>
                        </div>
                    </div>

                    <input id="custom-file-input" type="file" accept="image/*" style="display:none;" />

                    <div id="single-preview-wrap" class="single-preview-wrap">
                        <div class="single-preview-card">
                            <img id="single-preview-img" src="" alt="Preview">
                            <div class="preview-overlay-actions">
                                <button id="preview-upload-btn" class="preview-action-btn" title="Replace">Upload</button>
                                <button id="preview-clear-btn" class="preview-action-btn" title="Clear">Clear</button>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="hint-bar">
                    <b>Upload:</b> Click or drag to add a UI image &nbsp;&middot;&nbsp;
                    <b>Model:</b> Switch model tabs from the header &nbsp;&middot;&nbsp;
                    <kbd>Clear</kbd> removes the current image
                </div>

                <div class="examples-section">
                    <div class="examples-title">Quick Examples</div>
                    <div class="examples-scroll">
                        {EXAMPLE_CARDS_HTML}
                    </div>
                </div>
            </div>

            <div class="app-main-right">
                <div class="panel-card">
                    <div class="panel-card-title">Task Instruction</div>
                    <div class="panel-card-body">
                        <label class="modern-label" for="custom-query-input">Instruction Input</label>
                        <textarea id="custom-query-input" class="modern-textarea" rows="4" placeholder="e.g., click on the search bar, click on the model selector, click on the highlighted button..."></textarea>
                    </div>
                </div>

                <div style="padding:12px 20px;">
                    <button id="custom-run-btn" class="btn-run">
                        <span id="run-btn-label">Call CUA Agent</span>
                    </button>
                </div>

                <div class="annot-frame">
                    <div class="annot-title">Visualized Action Points</div>
                    <div class="annot-body">
                        <div id="annotated-output-placeholder" class="annot-placeholder">
                            {ANNOTATION_PLACEHOLDER_SVG}
                            <div class="annot-placeholder-title">Annotated UI preview will appear here</div>
                            <div class="annot-placeholder-sub">Detected click points and grounded actions will be drawn on the uploaded screenshot after inference.</div>
                        </div>
                        <img id="annotated-output-img" src="" alt="Annotated output">
                    </div>
                </div>

                <div class="output-frame">
                    <div class="out-title">
                        <span id="output-title-label">Agent Model Response</span>
                        <div class="out-title-right">
                            <button id="copy-output-btn" class="out-action-btn" title="Copy">{COPY_SVG} Copy</button>
                            <button id="save-output-btn" class="out-action-btn" title="Save">{SAVE_SVG} Save File</button>
                        </div>
                    </div>
                    <div class="out-body">
                        <div class="modern-loader" id="output-loader">
                            <div class="loader-spinner"></div>
                            <div class="loader-text">Running GUI agent...</div>
                            <div class="loader-bar-track"><div class="loader-bar-fill"></div></div>
                        </div>
                        <div class="output-scroll-wrap">
                            <textarea id="custom-output-textarea" class="output-textarea" placeholder="Agent response will appear here..." readonly></textarea>
                        </div>
                    </div>
                </div>

                <div class="settings-group">
                    <div class="settings-group-title">Advanced Settings</div>
                    <div class="settings-group-body">
                        <div class="slider-row">
                            <label>GPU Duration (seconds)</label>
                            <input type="range" id="custom-gpu-duration" min="60" max="300" step="30" value="60">
                            <span class="slider-val" id="custom-gpu-duration-val">60</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="exp-note">
            Experimental GUI Operator Suite &middot; Fara-7B, UI-TARS-1.5-7B, Holo2-4B, ActIO-UI-7B
        </div>

        <div class="app-statusbar">
            <div class="sb-section" id="sb-image-status">No image uploaded</div>
            <div class="sb-section sb-fixed" id="sb-run-state">Ready</div>
        </div>
    </div>
    """)

    run_btn = gr.Button("Run", elem_id="gradio-run-btn")

    demo.load(fn=noop, inputs=None, outputs=None, js=gallery_js)
    demo.load(fn=noop, inputs=None, outputs=None, js=wire_outputs_js)

    run_btn.click(
        fn=run_cua,
        inputs=[
            hidden_model_name,
            prompt,
            hidden_image_b64,
            gpu_duration_state,
        ],
        outputs=[result],
        js=r"""(m, p, img, gd) => {
            const modelEl = document.querySelector('.model-tab.active');
            const model = modelEl ? modelEl.getAttribute('data-model') : m;
            const promptEl = document.getElementById('custom-query-input');
            const promptVal = promptEl ? promptEl.value : p;
            const imgContainer = document.getElementById('hidden-image-b64');
            let imgVal = img;
            if (imgContainer) {
                const inner = imgContainer.querySelector('textarea, input');
                if (inner) imgVal = inner.value;
            }
            return [model, promptVal, imgVal, gd];
        }""",
    )

    example_load_btn.click(
        fn=load_example_data,
        inputs=[example_idx],
        outputs=[example_result],
        queue=False,
    )

if __name__ == "__main__":
    demo.queue(max_size=50).launch(
        css=css,
        mcp_server=True,
        ssr_mode=False,
        show_error=True,
        allowed_paths=["examples"],
    )
