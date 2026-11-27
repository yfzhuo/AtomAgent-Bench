"""Test the actual modified actions.py, not the previously suggested patch.

Default source: user's AtomWorldBench project.
Run: python3 test_current_delete_below.py
Optional: python3 test_current_delete_below.py --actions /absolute/path/actions.py

No API requests, dataset writes or execution of actions.py's __main__ block.
Requires numpy and ASE in the Python environment.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import sys
from io import BytesIO
import numpy as np
from ase import Atoms
from ase.io import read, write

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--actions', type=Path, default=Path(__file__).resolve().parents[1] / 'src/atom_world/actions.py')
args = parser.parse_args()
source = args.actions.resolve()
raw = source.read_bytes()
tree = ast.parse(raw, filename=str(source))
classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DeleteBelowAtomAction']
assert len(classes) == 1, 'Expected exactly one active DeleteBelowAtomAction'
compile(raw, str(source), 'exec')
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('actions_under_test', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # __name__ is not __main__: no example CIF writes.
Action = module.DeleteBelowAtomAction
assert Action.DEFAULT_Z_TOLERANCE == .1

cell = np.array([[4., 0., 0.], [1., 5., 0.], [.7, .4, 6.]])
heights = [.8, .9, .95, 1., 1.05, 1.1, 1.2]
atoms = Atoms('H7', positions=[[.2 + i*.1, .3, z] for i, z in enumerate(heights)], cell=cell, pbc=True)
atoms.set_tags(np.arange(7))
checks = 0
for include, expected in [(False, [1, 2, 3, 4, 5, 6]), (True, [6])]:
    action = Action(atoms.copy(), index=3, include_self=include)
    prompt = str(action)
    assert ('Including' if include else 'Excluding') in prompt
    assert '0.1 angstrom' in prompt
    assert 'Cartesian coordinate of the input structure' in prompt
    assert "reference atom's by at most 0.1 angstrom" in prompt
    assert 'Equivalently' not in prompt and 'z_center' not in prompt
    result = action.execute()
    assert result.get_tags().tolist() == expected
    np.testing.assert_array_equal(result.positions, atoms.positions[expected])
    np.testing.assert_array_equal(result.cell.array, atoms.cell.array)
    np.testing.assert_array_equal(result.pbc, atoms.pbc)
    # CIF export/read in memory only; verify real structure content, not just mask.
    buf = BytesIO()
    write(buf, result, format='cif', wrap=False)
    buf.seek(0)
    loaded = read(buf, format='cif')
    assert len(loaded) == len(result)
    np.testing.assert_allclose(loaded.cell.array, result.cell.array, atol=1e-8)
    np.testing.assert_allclose(loaded.positions, result.positions, atol=1e-8)
    print(('Including' if include else 'Excluding') + ': PASS; retained input indices ' + str(expected))
    checks += 1

# Independent scalar implementation: below-layer or (Including and in-layer).
for seed in range(100):
    rng = np.random.default_rng(seed)
    a = Atoms('H12', positions=rng.uniform(0, 3, (12, 3)), cell=cell, pbc=True)
    for include in (False, True):
        i = int(rng.integers(0, len(a)))
        center_z = a[i].position[2]
        expected = [j for j, atom in enumerate(a)
                    if atom.position[2] < center_z - .1
                    or (include and center_z - .1 <= atom.position[2] <= center_z + .1)]
        action = Action(a.copy(), i, include_self=include, z_tolerance=.1)
        assert np.flatnonzero(action._deletion_mask()).tolist() == expected
        checks += 1
    params = Action.randomize(a, rng=rng, config={'z_tolerance': .1})
    if params is not None:
        generated = Action(**params)
        count = int(np.count_nonzero(generated._deletion_mask()))
        assert 0 < count < len(a)
        assert params['z_tolerance'] == .1
        assert 0 < len(generated.execute()) < len(a)
        checks += 1

same_layer = Atoms('H2', positions=[[0, 0, 1], [1, 0, 1]], cell=cell, pbc=True)
assert Action.randomize(same_layer, rng=np.random.default_rng(3)) is None
for include in (False, True):
    action = Action(same_layer.copy(), 0, include_self=include)
    try:
        action.execute()
    except ValueError:
        pass
    else:
        raise AssertionError('Zero/all deletion must be rejected')
    checks += 1
for invalid in (-.1, float('nan'), float('inf')):
    try:
        Action(atoms.copy(), 3, z_tolerance=invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid tolerance accepted')
    checks += 1
# Custom tolerances must be reflected in both prompts and randomize output.
assert '0.05 angstrom' in str(Action(atoms.copy(), 3, z_tolerance=.05))
params = Action.randomize(atoms.copy(), rng=np.random.default_rng(4), config={'z_tolerance': .05})
assert params is not None and params['z_tolerance'] == .05
checks += 1
# Existing BaseAction helper must use the modified implementation correctly.
action, result = Action.apply_random(atoms, rng=np.random.default_rng(4), config={'z_tolerance': .1}, copy=True)
assert action is not None and 0 < len(result) < len(atoms)
np.testing.assert_array_equal(atoms.get_tags(), np.arange(7))
assert source.read_bytes() == raw, 'Source file changed during tests'
print('PASS: actual source class, boundaries, prompts, nonorthogonal cell, in-memory CIF export, independent-rule comparisons, randomize and apply_random.')
print(f'Test scenarios: {checks}; source SHA256: {hashlib.sha256(raw).hexdigest()}')
print('No model API calls; no source, datasets, references or saved experiment results modified.')
