"""POST the same texts to ort-server and tflite-server /classify and compare scores."""
import json
import sys
import time
import urllib.request

ORT, TFL = sys.argv[1], sys.argv[2]
TEXTS = [
    "My name is John Smith and my SSN is 123-45-6789.",
    "URGENT: verify your account at http://secure-login.example to avoid suspension.",
    "Thanks for the notes from today's standup, talk tomorrow.",
    "Your parcel could not be delivered. Pay the 1.99 EUR redelivery fee here: http://dhl-redelivery.example/pay",
    "Hi team, attached is the Q3 budget spreadsheet. Please review before Friday's meeting.",
    "Félicitations ! Vous avez gagné un iPhone 17 — cliquez ici pour réclamer votre prix 🎁",
    "ok",
    ("Dear customer, " + "we noticed unusual sign-in activity on your account and need you to confirm your identity. " * 60
     + "Click http://verify.example now."),
]


def post(base, text):
    req = urllib.request.Request(base + "/classify", data=json.dumps({"input": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.load(r)
    return body["labels"], (time.time() - t0) * 1000


worst = 0.0
for text in TEXTS:
    o, to = post(ORT, text)
    t, tt = post(TFL, text)
    delta = max(abs(o[k] - t.get(k, -1)) for k in o)
    worst = max(worst, delta)
    top_o = max(o, key=o.get)
    top_t = max(t, key=t.get)
    print(f"{'SAME' if top_o == top_t else 'DIFF'} top={top_t} ort={o[top_o]:.4f} tflite={t[top_t]:.4f} "
          f"delta={delta:.2e} ms ort={to:.0f} tflite={tt:.0f} | {text[:48]!r}")
print(f"max score delta across {len(TEXTS)} texts: {worst:.2e}")
