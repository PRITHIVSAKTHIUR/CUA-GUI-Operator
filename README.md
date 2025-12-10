# **CUA-GUI-Operator**

> A Gradio-based demonstration for Computer Use Agent (CUA) tasks, supporting multiple vision-language models: Microsoft Fara-7B, ByteDance UI-TARS-1.5-7B, Hcompany Holo2-4B, and Uniphore ActIO-UI-7B. Users upload UI screenshots (e.g., desktop or app interfaces), specify tasks (e.g., "Click on the search bar"), and receive parsed actions (e.g., clicks with coordinates) visualized as crosshairs and labels on the image. Handles JSON tool calls, coordinate normalization, and resizing for compatibility.

## Features

- **Multi-Model Support**: Switch between Fara-7B (JSON actions), UI-TARS-1.5-7B (localization), Holo2-4B (normalized coords), and ActIO-UI-7B (pyautogui-style actions).
- **Task-Driven Inference**: Natural language instructions generate structured outputs like clicks, types, with pixel or normalized coordinates.
- **Action Visualization**: Overlays crosshairs, circles, and labels (e.g., "Click" or "Type: 'Hello'") on the output image using PIL.
- **Response Parsing**: Model-specific regex/JSON extraction for actions; scales coordinates from resized inputs to original resolution.
- **Custom Theme**: OrangeRedTheme with gradients and enhanced typography for a professional UI.
- **Examples Integration**: Pre-loaded samples for quick testing across models.
- **Queueing Support**: Handles up to 50 concurrent inferences for efficient multi-user access.
- **Error Resilience**: Fallbacks for model loading; console logging for raw outputs and parsed actions.

## Example Preview

<img width="1804" height="1086" alt="Screenshot 2025-12-10 at 14-02-26 CUA GUI Operator - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/ca5d4338-db90-41a3-a51b-1dc208db7680" />

<img width="1816" height="1087" alt="Screenshot 2025-12-10 at 14-03-07 CUA GUI Operator - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/ea0a4912-af8b-4c8f-a1a9-b7ffb7f966c7" />

![image](https://github.com/user-attachments/assets/c4a4505f-613c-4fb5-87a9-d300c6c6fe06)

## Prerequisites

- Python 3.10 or higher.
- CUDA-compatible GPU (recommended for bfloat16/float16; falls back to CPU).
- Git for cloning dependencies.
- pip >= 23.0.0 (see pre-requirements.txt).
- Hugging Face account (optional, for model caching via `huggingface_hub`).

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/CUA-GUI-Operator.git
   cd CUA-GUI-Operator
   ```

2. Install pre-requirements (for pip version):
   Create a `pre-requirements.txt` file with the following content, then run:
   ```
   pip install -r pre-requirements.txt
   ```

   **pre-requirements.txt content:**
   ```
   pip>=23.0.0
   ```

3. Install dependencies:
   Create a `requirements.txt` file with the following content, then run:
   ```
   pip install -r requirements.txt
   ```

   **requirements.txt content:**
   ```
   transformers==4.57.1
   webdriver-manager
   huggingface_hub
   python-dotenv
   sentencepiece
   qwen-vl-utils
   gradio_modal
   torchvision
   matplotlib
   accelerate
   num2words
   pydantic
   requests
   pillow
   openai
   spaces
   einops
   torch
   peft
   ```

4. Start the application:
   ```
   python app.py
   ```
   The demo launches at `http://localhost:7860` (or the provided URL if using Spaces).

## Usage

1. **Select Model**: Choose from the radio buttons (default: Fara-7B).

2. **Upload Image**: Provide a UI screenshot (e.g., PNG of a desktop or app window; height up to 500px).

3. **Enter Task**: Describe the action (e.g., "Click on the Fara-7B model" or "Type 'Hello World' in the search box").

4. **Execute**: Click "Call CUA Agent" to run inference.

5. **View Results**:
   - Text: Raw model response with parsed actions.
   - Image: Annotated screenshot showing action points (crosshairs with labels).

### Example Workflow
- Select UI-TARS-1.5-7B.
- Upload a Hugging Face models page screenshot.
- Task: "Click on the VLMs Collection."
- Output: Response with click coordinates; image with red crosshair labeled "Click" on the element.

## Supported Models

| Model Name          | Key Capabilities                          | Coordinate Type | Notes                                      |
|---------------------|-------------------------------------------|-----------------|--------------------------------------------|
| Fara-7B            | JSON tool calls for actions (click, type) | Absolute pixels | Microsoft; supports text input in actions. |
| UI-TARS-1.5-7B    | Element localization prompts             | Absolute pixels | ByteDance; resizes images for processing.  |
| Holo2-4B           | JSON schema for normalized coords        | Normalized (0-1000) | Hcompany; scales to original resolution.   |
| ActIO-UI-7B        | Pyautogui-style actions (click(x,y))     | Absolute pixels | Uniphore; verbose outputs up to 1024 tokens.|

## Troubleshooting

- **Model Loading Errors**: Verify transformers 4.57.1; check CUDA with `torch.cuda.is_available()`. Use `torch.float32` if bfloat16/float16 OOM occurs.
- **No Actions Parsed**: Ensure task specificity; raw output in console. Increase max_new_tokens for verbose models like ActIO.
- **Coordinate Scaling Issues**: Holo2/UI-TARS resize internally; scaling applied automatically. Test with examples.
- **Visualization Errors**: PIL font fallback used; ensure images are RGB numpy arrays.
- **Queue Full**: Increase `max_size` in `demo.queue()` for higher traffic.
- **Vision Utils**: Install `qwen-vl-utils` for Fara/ActIO processing; smart_resize handles min/max pixels.
- **UI Rendering**: Set `ssr_mode=True` if gradients fail; check CSS for container styles.

## Contributing

Contributions encouraged! Fork, create a feature branch (e.g., for new models), and submit PRs with tests. Focus areas:
- Additional CUA models or action types (e.g., scroll, drag).
- Batch processing or video frame support.
- Enhanced parsing for edge cases.

Repository: [https://github.com/PRITHIVSAKTHIUR/CUA-GUI-Operator.git](https://github.com/PRITHIVSAKTHIUR/CUA-GUI-Operator.git)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

Built by [Prithiv Sakthi](https://huggingface.co/prithivMLmods). Report issues via the repository.
