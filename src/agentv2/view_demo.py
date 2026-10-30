import os, base64, argparse, json, importlib

def load_init_cif(action, sample):
    path = f"src/data/{action}_qa_format.json"
    with open(path) as f:
        data = json.load(f)
    item = data[sample]
    for k in ("input_cif", "cif", "init_cif", "structure", "input_structure",
              "input", "cif_str", "initial_cif"):
        if k in item and isinstance(item[k], str) and "data_" in item[k]:
            return item[k]
    raise KeyError(f"找不到 CIF 字段, 该记录的字段有: {list(item.keys())}")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="insert_between_atoms_action")
    ap.add_argument("--sample", type=int, default=13)
    ap.add_argument("--highlight", type=int, nargs="*", default=[8, 16],
                    help="要高亮的原子序号")
    ap.add_argument("--out", default="view_out")
    args = ap.parse_args()

    from src.agent.structure_env import StructureEnvironment
    init_cif = load_init_cif(args.action, args.sample)
    env = StructureEnvironment(init_cif)

    os.makedirs(args.out, exist_ok=True)
    angles = [("default", None, None), ("axis_a", 0, 0),
              ("azim45", 45, 15), ("azim90", 90, 20), ("azim135", 135, 30)]
    for name, azim, elev in angles:
        obs = env.view(target="current", azim=azim, elev=elev,
                       show_indices=True, highlight=args.highlight)
        b64 = obs["image_base64"]
        fn = os.path.join(args.out, f"s{args.sample}_{name}.png")
        with open(fn, "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"saved {fn}  (occluded 的原子应为虚线框)")

if __name__ == "__main__":
    main()
