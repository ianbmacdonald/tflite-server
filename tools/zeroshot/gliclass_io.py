# gliclass_io.py prep <model_dir> <out_dir> | prep-ids <tokenizer.json> <out_dir> | decide <logits.f32>
# Own re-implementation of the model card's host steps (linearize, tokenize, fp16 table lookup,
# label routing; softmax / sigmoid); it does not import the repository's Python.
import sys, json, numpy as np
TEXT = ("The Zorvik X2 router from Tallowmere Networks keeps dropping the connection every evening "
        "around 9 pm. I have restarted it and reset it to factory settings, but the status light "
        "still blinks orange.")
LABELS = ["connectivity problem", "hardware defect", "billing issue", "feature request", "installation help"]
S, SLOTS, VOCAB, HID = 128, 25, 50370, 384
if sys.argv[1] == "prep":
    from tokenizers import Tokenizer
    md, out = sys.argv[2], sys.argv[3]
    tok = Tokenizer.from_file(f"{md}/host_assets/tokenizer.json"); tok.no_truncation(); tok.no_padding()
    lab, pad = tok.token_to_id("<<LABEL>>"), tok.token_to_id("[PAD]")
    ids = tok.encode("".join("<<LABEL>>" + l for l in LABELS) + "<<SEP>>" + TEXT).ids
    pos = [i for i, t in enumerate(ids) if t == lab]
    assert len(pos) == len(LABELS) and len(ids) <= S, (len(pos), len(ids))
    table = np.memmap(f"{md}/host_assets/tok_embeddings_fp16.bin", dtype="<f2", mode="r", shape=(VOCAB, HID))
    p = np.full(S, pad, np.int64); p[:len(ids)] = ids
    table[p].astype(np.float32)[None].tofile(f"{out}/embeds.f32")
    m = np.zeros((1, S), np.float32); m[0, :len(ids)] = 1; m.tofile(f"{out}/mask.f32")
    r = np.zeros((1, SLOTS, S), np.float32)
    for k, q in enumerate(pos): r[0, k, q] = 1
    r.tofile(f"{out}/routing.f32")
    print("tokens", len(ids), "label positions", pos)
elif sys.argv[1] == "prep-ids":
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(sys.argv[2]); tok.no_truncation(); tok.no_padding(); out = sys.argv[3]
    lab, pad = tok.token_to_id("<<LABEL>>"), tok.token_to_id("[PAD]")
    ids = tok.encode("".join("<<LABEL>>" + l for l in LABELS) + "<<SEP>>" + TEXT).ids
    pos = [i for i, t in enumerate(ids) if t == lab]
    assert len(pos) == len(LABELS) and len(ids) <= S, (len(pos), len(ids))
    p = np.full((1, S), pad, np.int32); p[0, :len(ids)] = ids; p.tofile(f"{out}/ids.i32")
    m = np.zeros((1, S), np.int32); m[0, :len(ids)] = 1; m.tofile(f"{out}/mask.i32")
    r = np.zeros((1, SLOTS, S), np.float32)
    for k, q in enumerate(pos): r[0, k, q] = 1
    r.tofile(f"{out}/routing.f32")
    print("tokens", len(ids), "label id", lab, "label positions", pos)
else:
    z = np.fromfile(sys.argv[2], dtype=np.float32)[:len(LABELS)]
    e = np.exp(z - z.max()); sm = e / e.sum(); sg = 1 / (1 + np.exp(-z))
    print(json.dumps({"logits": [round(float(x), 4) for x in z],
                      "single": {LABELS[int(sm.argmax())]: round(float(sm.max()), 3)},
                      "multi": {l: round(float(s), 3) for l, s in zip(LABELS, sg) if s >= 0.5}}))
