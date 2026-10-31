# gmb_torch_ref.py <hf_dir> <in_dir> <out.f32>: the export's torch module on the prepared inputs (reference logits).
import sys, runpy, numpy as np, torch
hf, ind, out = sys.argv[1:4]
sys.argv = ["x", hf, "/dev/null"]
src = open(__file__.replace("gmb_torch_ref.py", "export_gliclass_modern.py")).read().split("litert_torch.convert")[0]
g = {}
exec(compile(src, "export_gliclass_modern.py", "exec"), g)
m = g["m"]
ids = torch.from_numpy(np.fromfile(f"{ind}/ids.i32", np.int32).reshape(1, 128))
mask = torch.from_numpy(np.fromfile(f"{ind}/mask.i32", np.int32).reshape(1, 128))
r = torch.from_numpy(np.fromfile(f"{ind}/routing.f32", np.float32).reshape(1, 25, 128))
with torch.no_grad():
    m(ids, mask, r).numpy().astype(np.float32).tofile(out)
print("wrote", out)
