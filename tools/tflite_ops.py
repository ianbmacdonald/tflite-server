"""Write the TFLite builtin-op list a set of .tflite models needs, for -DTFLITE_SERVER_OPS=<file>.

    python tools/tflite_ops.py model.tflite [more.tflite ...] > ops.txt

Every operator code of every model is listed (the interpreter resolves all of them at load, even ops the
XNNPACK delegate later takes over), one BuiltinOperator name per line; custom ops as CUSTOM:<name>.
Needs ai-edge-litert (pip install ai-edge-litert) for the flatbuffer schema.
"""

import sys
from pathlib import Path

from ai_edge_litert import schema_py_generated as schema


def model_ops(path):
    m = schema.Model.GetRootAsModel(Path(path).read_bytes(), 0)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items() if not k.startswith("_")}
    used = {m.Subgraphs(s).Operators(i).OpcodeIndex()
            for s in range(m.SubgraphsLength()) for i in range(m.Subgraphs(s).OperatorsLength())}
    ops = set()
    for i in used:
        c = m.OperatorCodes(i)
        code = max(c.BuiltinCode(), c.DeprecatedBuiltinCode())
        if names[code] == "CUSTOM":
            ops.add("CUSTOM:" + c.CustomCode().decode())
        else:
            ops.add(names[code])
    return ops


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    per = {p: model_ops(p) for p in sys.argv[1:]}
    for p, ops in per.items():
        print(f"# {Path(p).name}: {len(ops)} ops")
    for op in sorted(set().union(*per.values())):
        print(op)


if __name__ == "__main__":
    main()
