"""Captions for Time Cube: what the AI sees in a video, and answers about a moment.

The model is Qwen3-VL 4B as the single file ComfyUI uses (qwen3vl_4b_fp8_scaled.safetensors). It is looked for where
the TIME_CUBE_QWEN_MODEL environment variable points (the file itself, or a folder that holds it, such as ComfyUI's
models\\text_encoders), then in this app's models folder. Nothing is downloaded. Qwen3-VL needs transformers 4.57 or
newer, so this runs in its own small environment (%LOCALAPPDATA%\\TimeCube\\caption-env: the system's packages
plus that transformers; made by make_caption_env.ps1). The server keeps it running while the section is used and
closes it when idle.

    python caption.py --serve [--parent <server pid>]

One JSON command per stdin line, JSON lines back (a describe also reports progress):
    {"cmd": "describe", "src", "times": [seconds, ...]}  -> {"t": "result", "moments": [{"time", "text"}], "place"}
    {"cmd": "ask", "src", "time", "question"}             -> {"t": "result", "answer"}

ComfyUI keeps the language model's big weights as fp8 with one scale each; here they stay fp8 on the GPU and are
unpacked layer by layer as they are used (about 5.5 GB of graphics memory in all).
"""
import argparse
import io
import json
import os
import struct
import subprocess
import sys
import time

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
from server import NO_WINDOW, find_tool, no_throttle  # noqa: E402

TOKENIZER = os.path.join(APP_DIR, "models", "qwen-tokenizer")
# the model file: where TIME_CUBE_QWEN_MODEL points (the file, or a folder holding it), else this app's models folder
MODEL_FILE = "qwen3vl_4b_fp8_scaled.safetensors"
MODEL_ENV = "TIME_CUBE_QWEN_MODEL"
MODEL_DIR = os.path.join(APP_DIR, "models")
FRAME_LONG = 640            # frames are looked at this big (long side)
NEEDS_GB = 6.0              # graphics memory the model needs; with less free, Windows swaps and it crawls
STOP = [151645, 151643]     # <|im_end|>, <|endoftext|>
PROMPT_MOMENT = ("Describe this moment of a video in one sentence of about 20 words: who or what is there, what is "
                 "happening, and the setting. No preamble.")
PROMPT_PLACE = ("Where might this be filmed? Give the kind of place and, if you can tell, the city or country. "
                "Answer in one short line, as a guess.")
CHAT = ("{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}"
        "{{ message['content'] }}{% else %}{% for c in message['content'] %}{% if c['type'] == 'image' %}"
        "<|vision_start|><|image_pad|><|vision_end|>{% elif c['type'] == 'text' %}{{ c['text'] }}{% endif %}"
        "{% endfor %}{% endif %}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def say(**msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def model_path():
    """The Qwen3-VL 4B file, or None when it isn't found."""
    env = os.environ.get(MODEL_ENV, "").strip().strip('"')
    places = []
    if env:
        places += [os.path.join(env, MODEL_FILE)] if os.path.isdir(env) else [env]
    places.append(os.path.join(MODEL_DIR, MODEL_FILE))
    for p in places:
        if os.path.isfile(p):
            return p
    return None


class Captioner:
    def __init__(self):
        self.model = self.proc = self.tok = None

    def load(self):
        if self.model is not None:
            return
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import transformers
        from accelerate import init_empty_weights
        from safetensors import safe_open
        if tuple(int(x) for x in transformers.__version__.split(".")[:2]) < (4, 57):
            raise RuntimeError("captions need transformers 4.57 or newer: run make_caption_env.ps1")
        from transformers import (Qwen2TokenizerFast, Qwen2VLImageProcessorFast, Qwen3VLConfig,
                                  Qwen3VLForConditionalGeneration, Qwen3VLProcessor)
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
        if not torch.cuda.is_available():
            raise RuntimeError("captions need the NVIDIA graphics card, and PyTorch can't see it")
        free = torch.cuda.mem_get_info()[0] / 1e9
        if free < NEEDS_GB:
            raise RuntimeError(f"the graphics card has only {free:.1f} GB free and captions need about {NEEDS_GB:.0f} GB:"
                               " close other programs using it (ComfyUI, other AI jobs) and try again")
        path = model_path()
        if not path:
            raise RuntimeError(f"the Qwen3-VL 4B model ({MODEL_FILE}) wasn't found: put it in the app's models folder or"
                               f" set {MODEL_ENV} to the file (see the README)")

        class FP8Linear(nn.Module):
            """A Linear whose weight stays fp8 on the GPU, unpacked (times its scale) for each use."""

            def __init__(self, lin):
                super().__init__()
                self.in_features, self.out_features = lin.in_features, lin.out_features
                self.weight = self.scale = self.bias = None

            def forward(self, x):
                return F.linear(x, self.weight.to(x.dtype) * self.scale.to(x.dtype), self.bias)

        # Qwen3-VL-4B's settings (the same as ComfyUI's qwen3vl.py / llama.py)
        cfg = Qwen3VLConfig(
            text_config=dict(vocab_size=151936, hidden_size=2560, intermediate_size=9728, num_hidden_layers=36,
                             num_attention_heads=32, num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-6,
                             rope_theta=5000000.0, max_position_embeddings=262144, hidden_act="silu",
                             attention_bias=False, tie_word_embeddings=True, dtype="bfloat16",
                             rope_scaling={"rope_type": "default", "mrope_section": [24, 20, 20], "mrope_interleaved": True}),
            vision_config=dict(depth=24, hidden_size=1024, intermediate_size=4096, num_heads=16, in_channels=3,
                               patch_size=16, spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=2560,
                               num_position_embeddings=2304, deepstack_visual_indexes=[5, 11, 17],
                               hidden_act="gelu_pytorch_tanh"),
            image_token_id=151655, video_token_id=151656, vision_start_token_id=151652, vision_end_token_id=151653,
            tie_word_embeddings=True)
        for c in (cfg, cfg.text_config, cfg.vision_config):
            c._attn_implementation = "sdpa"
        with init_empty_weights():
            model = Qwen3VLForConditionalGeneration(cfg)
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        header.pop("__metadata__", None)

        def to_hf(k):   # ComfyUI's names -> transformers'
            if k.startswith("model.") and not k.startswith("model.visual."):
                return "model.language_model." + k[len("model."):]
            return k

        for k, v in header.items():
            if v["dtype"] == "F8_E4M3":
                parent, attr = to_hf(k[:-len(".weight")]).rsplit(".", 1)
                mod = model.get_submodule(parent)
                setattr(mod, attr, FP8Linear(getattr(mod, attr)))
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith((".comfy_quant", ".input_scale")):
                    continue
                t = f.get_tensor(k)
                if k.endswith(".weight_scale"):
                    model.get_submodule(to_hf(k[:-len(".weight_scale")])).scale = t.to("cuda", torch.float32)
                    continue
                parent, attr = to_hf(k).rsplit(".", 1)
                mod = model.get_submodule(parent)
                if isinstance(mod, FP8Linear):
                    setattr(mod, attr, t.to("cuda") if attr == "weight" else t.to("cuda", torch.bfloat16))
                else:
                    setattr(mod, attr, nn.Parameter(t.to("cuda", torch.bfloat16), requires_grad=False))
        model.lm_head.weight = model.model.language_model.embed_tokens.weight     # the 4B ties them
        # the rotary tables were made on the meta device: make them again for real
        model.model.language_model.rotary_emb.__init__(cfg.text_config, device="cuda")
        vis = model.model.visual
        vis.rotary_pos_emb.__init__(cfg.vision_config.hidden_size // cfg.vision_config.num_heads // 2)
        vis.rotary_pos_emb.to("cuda")
        model.generation_config.eos_token_id = STOP
        model.generation_config.pad_token_id = STOP[1]
        self.model = model.eval()
        self.tok = Qwen2TokenizerFast.from_pretrained(TOKENIZER)
        ip = Qwen2VLImageProcessorFast(patch_size=16, merge_size=2, temporal_patch_size=2, image_mean=[0.5] * 3,
                                       image_std=[0.5] * 3,
                                       size={"shortest_edge": 64 * 32 * 32, "longest_edge": 400 * 32 * 32})
        self.proc = Qwen3VLProcessor(image_processor=ip, tokenizer=self.tok, video_processor=Qwen3VLVideoProcessor(),
                                     chat_template=CHAT)

    def answer(self, img, question, max_new=90):
        import torch
        self.load()
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = self.proc(text=[text], images=[img], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.05)
        return self.tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


C = Captioner()


def frame(src, t):
    from PIL import Image
    ff = find_tool("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg was not found on this computer")
    r = subprocess.run([ff, "-v", "error", "-nostdin", "-ss", f"{max(0.0, float(t)):.3f}", "-i", src, "-frames:v", "1",
                        "-vf", f"scale='if(gt(iw,ih),{FRAME_LONG},-2)':'if(gt(iw,ih),-2,{FRAME_LONG})'",
                        "-f", "image2pipe", "-c:v", "png", "pipe:1"],
                       capture_output=True, stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError("that moment of the video can't be read")
    return Image.open(io.BytesIO(r.stdout)).convert("RGB")


def describe(src, times):
    t0 = time.time()
    C.load()
    moments = []
    for i, t in enumerate(times):
        moments.append({"time": round(float(t), 2), "text": C.answer(frame(src, t), PROMPT_MOMENT, 60)})
        say(t="progress", done=i + 1, total=len(times) + 1)
    place = C.answer(frame(src, times[len(times) // 2]), PROMPT_PLACE, 30)
    say(t="progress", done=len(times) + 1, total=len(times) + 1)
    return {"moments": moments, "place": place, "seconds": round(time.time() - t0, 1)}


def ask(src, t, question):
    q = str(question).strip()[:400]
    if not q:
        raise ValueError("type a question")
    return {"answer": C.answer(frame(src, t), q + " Answer briefly.", 120)}


def serve():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
            if c.get("cmd") == "describe":
                say(t="result", **describe(c["src"], c["times"]))
            elif c.get("cmd") == "ask":
                say(t="result", **ask(c["src"], c.get("time", 0), c.get("question", "")))
            elif c.get("cmd") == "ping":
                C.load()
                say(t="result", ok=True)
            else:
                raise ValueError(f"unknown command {c.get('cmd')!r}")
        except Exception as e:  # noqa: BLE001
            say(t="error", error=str(e) or e.__class__.__name__)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--parent", type=int)
    a = ap.parse_args()
    if a.parent:
        from depth import watch_parent
        watch_parent(a.parent)
    no_throttle()
    if a.serve:
        serve()


if __name__ == "__main__":
    main()
