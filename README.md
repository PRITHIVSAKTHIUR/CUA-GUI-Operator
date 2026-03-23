# **CUA-GUI-Operator**

CUA-GUI-Operator is an experimental, advanced computer-use agent (CUA) and visual action grounding suite designed to bridge the gap between natural language instructions and graphical user interface (GUI) interactions. By leveraging a specialized roster of vision-language models—including Microsoft's Fara-7B, ByteDance's UI-TARS-1.5, Holo2-4B, and ActIO-UI-7B—this application can accurately localize UI elements, predict the next logical agentic step, and ground actions directly onto interface screenshots. The suite features a bespoke, highly interactive web frontend engineered with custom HTML, CSS, and JavaScript to support seamless drag-and-drop workflows. Upon receiving a task instruction and a screenshot, the operator not only generates the raw model response but also renders a visualized, annotated image highlighting the exact interaction coordinates. Fully GPU-accelerated, CUA-GUI-Operator provides researchers and developers with a powerful sandbox for testing and deploying intelligent, autonomous GUI agents.

<img width="1920" height="2020" alt="Screenshot 2026-03-23 at 17-12-23 CUA GUI Operator - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/a5122619-f355-49a9-950b-a86e21a13286" />

### **Key Features**

* **Multi-Model CUA Architecture:** Seamlessly switch between state-of-the-art GUI agent models directly from the interface. Supported models include `Fara-7B`, `UI-TARS-1.5-7B`, `Holo2-4B`, and `ActIO-UI-7B`.
* **Visual Action Grounding:** Automatically parses the model's coordinate output and renders an annotated preview image, visually highlighting the precise points of interaction (e.g., clicks, bounding boxes) on the uploaded screenshot.
* **Custom User Interface:** Features a bespoke, responsive Gradio frontend. It includes a drag-and-drop media zone, real-time output streaming, and an interactive annotation display layer.
* **Output Management:** Built-in actions allow users to instantly copy the raw output text to their clipboard or save the generated agent response directly as a `.txt` file.
* **Hardware Optimization:** Utilizes dynamic memory management and garbage collection to run intensive 7B parameter models smoothly on compatible CUDA-enabled GPUs.

### **Repository Structure**

```text
├── examples/
│   ├── 1.png
│   ├── 2.png
│   └── 3.png
├── app.py
├── LICENSE
├── pre-requirements.txt
├── README.md
└── requirements.txt
```

### **Installation and Requirements**

To run CUA-GUI-Operator locally, you need to configure a Python environment with the following dependencies. Ensure you have a compatible CUDA-enabled GPU for optimal performance.

**1. Install Pre-requirements**
Run the following command to update pip to the required version:
```bash
pip install pip>=23.0.0
```

**2. Install Core Requirements**
Install the necessary machine learning and UI libraries. You can place these in a `requirements.txt` file and run `pip install -r requirements.txt`.

```text
git+https://github.com/huggingface/transformers.git@v4.57.1
huggingface_hub
python-dotenv
sentencepiece
qwen-vl-utils
torch==2.8.0
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
peft
```

### **Usage**

Once your environment is set up and the dependencies are installed, you can launch the application by running the main Python script:

```bash
python app.py
```

After the script initializes the interface, it will provide a local web address (usually `http://127.0.0.1:7860/`) which you can open in your browser to interact with the models. Note that the selected models will be downloaded and loaded into VRAM upon their first invocation. Provide a specific instruction (e.g., "Click on the search bar") alongside your UI screenshot to see the grounded interaction.

### **License and Source**

* **License:** Apache License - Version 2.0
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/CUA-GUI-Operator.git](https://github.com/PRITHIVSAKTHIUR/CUA-GUI-Operator.git)
