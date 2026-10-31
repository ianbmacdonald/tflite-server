# export_gliclass_modern.py <hf_dir> <out.tflite>
# Exports knowledgator/gliclass-modern-base-v3.0 (uni-encoder, pooling "first", scorer "simple")
# to TFLite with litert-torch. The GLiClass head is re-implemented here from the checkpoint's
# tensor names (text_projector / classes_projector: Linear-GELU-Linear; scorer: dot product of
# the projected [CLS] state with each projected <<LABEL>> state); the gliclass package is not used.
# Graph: input_ids [1,S] int32, attention_mask [1,S] int32, label_routing [1,25,S] f32 -> logits [1,25].
import json, sys
import torch
from safetensors.torch import load_file
from transformers import ModernBertConfig, ModernBertModel
import litert_torch

hf, out = sys.argv[1], sys.argv[2]
S = 128
cfg = json.load(open(f"{hf}/config.json"))
ec = ModernBertConfig(**{k: v for k, v in cfg["encoder_config"].items() if not k.startswith("_")})
ec._attn_implementation = "eager"
ec.vocab_size = cfg["vocab_size"]


class Proj(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear_1 = torch.nn.Linear(d, d)
        self.linear_2 = torch.nn.Linear(d, d)

    def forward(self, x):
        return self.linear_2(torch.nn.functional.gelu(self.linear_1(x)))


class GLiClassUni(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_model = ModernBertModel(ec)
        self.text_projector = Proj(cfg["hidden_size"])
        self.classes_projector = Proj(cfg["hidden_size"])
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, input_ids, attention_mask, label_routing):
        h = self.encoder_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text = self.text_projector(h[:, 0])                      # [1,D]
        classes = self.classes_projector(torch.bmm(label_routing, h))  # [1,25,D]
        return torch.bmm(classes, text.unsqueeze(-1)).squeeze(-1)  # [1,25]


m = GLiClassUni()
sd = {k[len("model."):]: v for k, v in load_file(f"{hf}/model.safetensors").items()}
missing, unexpected = m.load_state_dict(sd, strict=False)
print("missing", missing, "unexpected", unexpected)
assert not unexpected and all("rotary" in k or k == "encoder_model.embeddings.norm.bias" for k in missing), missing
m.eval()
args = (torch.zeros(1, S, dtype=torch.int32), torch.ones(1, S, dtype=torch.int32), torch.zeros(1, 25, S))
with torch.no_grad():
    print("torch logits shape", m(*args).shape)
litert_torch.convert(m, args).export(out)
print("wrote", out)
