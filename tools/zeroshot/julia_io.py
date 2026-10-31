# julia_io.py prep <model_dir> <out_dir> | decide <out_dir> <token_logits.f32>
# Own implementation of the Julia-1-LiteRT README's host contract (it does not import julia_litert.py):
# <bos> "choice question: {q}" <eos> <mask> " opt0" <mask> " opt1" ... <eos> state <eos>,
# fp16 table lookup, attention mask, qtype one-hot; softmax over the logits at the <mask> positions.
import json, sys
import numpy as np
Q = "What is the cause of this ticket?"
OPTS = ["connectivity problem", "hardware defect", "billing issue", "feature request", "installation help"]
STATE = ("The Zorvik X2 router from Tallowmere Networks keeps dropping the connection every evening "
         "around 9 pm. I have restarted it and reset it to factory settings, but the status light "
         "still blinks orange.")
S, HID, VOCAB = 512, 384, 256000
if sys.argv[1] == "prep":
    from tokenizers import Tokenizer
    md, out = sys.argv[2], sys.argv[3]
    tok = Tokenizer.from_file(f"{md}/tokenizer.json")
    tid = {t: tok.token_to_id(t) for t in ("<bos>", "<eos>", "<mask>")}
    assert all(v is not None for v in tid.values()), tid
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    ids = [tid["<bos>"]] + enc(f"choice question: {Q}") + [tid["<eos>"]]
    pos = []
    for o in OPTS:
        pos.append(len(ids)); ids += [tid["<mask>"]] + enc(" " + o)
    ids += [tid["<eos>"]] + enc(STATE) + [tid["<eos>"]]
    assert len(ids) <= S
    t = np.memmap(f"{md}/julia1_token_table_fp16.bin", dtype="<f2", mode="r", shape=(VOCAB, HID))
    p = np.zeros(S, np.int64); p[:len(ids)] = ids
    t[p].astype(np.float32)[None].tofile(f"{out}/embeds.f32")
    m = np.zeros((1, S), np.float32); m[0, :len(ids)] = 1; m.tofile(f"{out}/mask.f32")
    np.array([[1, 0, 0]], np.float32).tofile(f"{out}/qtype.f32")
    json.dump(pos, open(f"{out}/positions.json", "w"))
    print("tokens", len(ids), "mask positions", pos)
else:
    pos = json.load(open(f"{sys.argv[2]}/positions.json"))
    z = np.fromfile(sys.argv[3], dtype=np.float32)[pos]
    e = np.exp(z - z.max()); p = e / e.sum()
    print(json.dumps({"choice": OPTS[int(p.argmax())], "p": {o: round(float(x), 3) for o, x in zip(OPTS, p)}}))
