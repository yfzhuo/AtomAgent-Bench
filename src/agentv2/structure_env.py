import io
import base64
import contextlib
import traceback

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ase.visualize.plot import Matplotlib
from pymatgen.core import Structure, Element, Lattice
from pymatgen.io.ase import AseAtomsAdaptor


class StructureEnvironment:
    def __init__(self, init_cif: str, max_pixels: int = 600,
                 allow_imaging: bool = True):
        self.input_structure = Structure.from_str(init_cif, fmt="cif")
        self.structure = self.input_structure.copy()
        self.azim = 20.0
        self.elev = 10.0
        self.max_pixels = max_pixels
        self.allow_imaging = allow_imaging
        self.view_calls = 0
        self.code_calls = 0

    def _render(self, structure, show_indices=False, highlight=None):
        atoms = AseAtomsAdaptor.get_atoms(structure)
        rotation = f'{self.elev}x,{self.azim}y,0z'
        fig, ax = plt.subplots(figsize=(5, 5))
        plotter = Matplotlib(atoms, ax, rotation=rotation)
        plotter.write()
        ax.axis('off')

        pos2d = np.asarray(plotter.positions)[:, :2]
        n_atoms = len(structure)
        if show_indices:
            import matplotlib.patheffects as pe
            for i, (px, py) in enumerate(pos2d[:n_atoms]):
                ax.text(px, py, str(i), fontsize=7, color='black',
                        ha='center', va='center', zorder=10, fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=2.0,
                                                    foreground='white')])
        if highlight:
            n = n_atoms
            pos3d = np.asarray(plotter.positions)[:n]
            depth = pos3d[:, 2]
            radii = np.asarray(plotter.d)[:n] / 2.0

            for i in highlight:
                if not (0 <= i < n):
                    continue

                occluded = False
                for j in range(n):
                    if j == i:
                        continue
                    if depth[j] <= depth[i]:
                        continue
                    d2d = np.hypot(pos2d[j, 0] - pos2d[i, 0],
                                   pos2d[j, 1] - pos2d[i, 1])
                    if d2d < radii[j] * 0.8:
                        occluded = True
                        break
                r = radii[i] if radii[i] > 0 else 0.4
                if occluded:
                    ax.add_patch(plt.Circle(
                        (pos2d[i, 0], pos2d[i, 1]), r * 1.0,
                        fill=False, linestyle='--', linewidth=2.0,
                        edgecolor='red', zorder=12))
                else:
                    ax.add_patch(plt.Circle(
                        (pos2d[i, 0], pos2d[i, 1]), r * 1.0,
                        fill=False, linestyle='-', linewidth=2.0,
                        edgecolor='red', zorder=12))
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)

        data = buf.getvalue()
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(data))
            w, h = img.size
            s = self.max_pixels / max(w, h)
            if s < 1:
                img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
            out = io.BytesIO()
            img.save(out, format='PNG', optimize=True)
            data = out.getvalue()
        except ImportError:
            pass
        return base64.b64encode(data).decode('utf-8')

    def view(self, target="current", axis=None, azim=None, elev=None,
             show_indices=False, highlight=None):
        self.view_calls += 1
        if target == "input":
            st = self.input_structure
        elif target == "current":
            st = self.structure
        else:
            return self._obs(f"unknown target '{target}', use 'current'/'input'",
                             ok=False)

        presets = {'a': (0.0, 0.0), 'b': (90.0, 0.0), 'c': (0.0, 90.0)}
        if axis in presets:
            self.azim, self.elev = presets[axis]
        if azim is not None:
            self.azim = float(azim)
        if elev is not None:
            self.elev = float(elev)

        info = (f"target={target}, n_atoms={len(st)}, "
                f"formula={st.composition.formula}, "
                f"camera(azim={self.azim}, elev={self.elev})")

        if not self.allow_imaging:
            return self._obs(f"[imaging disabled] {info}", ok=True, st=st)

        img = self._render(st, show_indices=show_indices,
                           highlight=highlight or [])
        return self._obs(f"rendered view: {info}", ok=True, image_b64=img, st=st)

    def run_python(self, code: str):
        self.code_calls += 1
        if not code or not isinstance(code, str):
            return self._obs("empty code", ok=False)

        ns = {
            "__builtins__": __builtins__,
            "np": np,
            "Structure": Structure,
            "Element": Element,
            "Lattice": Lattice,
            "AseAtomsAdaptor": AseAtomsAdaptor,
            "structure": self.structure.copy(),
            "input_cif": self.input_structure.to(fmt="cif"),
        }
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                exec(code, ns)
        except Exception:
            return self._obs("code raised an exception (structure unchanged)",
                             ok=False,
                             stdout=out.getvalue() + "\n" + traceback.format_exc())

        new_st = ns.get("structure")
        if not isinstance(new_st, Structure):
            return self._obs("after running, variable `structure` must still be a "
                             "pymatgen.Structure (structure unchanged)",
                             ok=False, stdout=out.getvalue())
        self.structure = new_st
        return self._obs("code executed; current structure updated",
                         ok=True, stdout=out.getvalue())

    def export_cif(self) -> str:
        if len(self.structure) == 0:
            raise ValueError(
                "Cannot export CIF: structure has 0 atoms "
                "(all sites were deleted). Please undo the last delete "
                "or re-load the initial structure from `input_cif`."
            )
        try:
            return self.structure.to(fmt="cif")
        except Exception as e:
            raise ValueError(
                f"Cannot export CIF: pymatgen serialization failed "
                f"({type(e).__name__}: {e}). Current structure has "
                f"{len(self.structure)} atoms, formula="
                f"{self.structure.composition.formula}."
            ) from e
    def _obs(self, message, ok=True, image_b64=None, stdout=None, st=None):
        st = st if st is not None else self.structure
        return {
            "ok": ok,
            "message": message,
            "n_atoms": len(st),
            "formula": st.composition.formula,
            "stdout": (stdout or "")[:4000],
            "image_base64": image_b64,
            "counts": {"view": self.view_calls, "code": self.code_calls},
        }
