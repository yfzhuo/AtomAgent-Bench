import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from abc import ABC, abstractmethod
DEFAULT_CONFIG = {'radius_min': 1.0, 'radius_max': 4.0, 'distance_min': 0.1, 'distance_max': 3.0, 'dpos_scale': 2.0, 'distance_ratio_min': 0.1, 'distance_ratio_max': 0.9, 'angle_min': 45.0, 'angle_max': 315.0, 'symbol_pool': list(chemical_symbols[1:]), 'decimal_places': 3}

def _get_config(config=None):
    """Merge user config with defaults."""
    if config is None:
        return dict(DEFAULT_CONFIG)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config)
    return cfg

def _safe_radius(atoms, index, rmin, rmax):
    """Adjust radius bounds to avoid deleting all atoms."""
    if index is not None and len(atoms) > 1:
        distances = atoms.get_distances(index, range(len(atoms)), mic=True)
        nonzero = distances[distances > 0]
        if len(nonzero) > 0:
            max_dist = np.max(nonzero)
            min_dist = np.min(nonzero)
            rmax = min(rmax, max_dist * 0.99)
            rmin = max(rmin, min_dist * 1.01)
            if rmax < rmin:
                rmax = max_dist * 0.5
            if rmax < 0.1:
                rmax = 0.1
    return (rmin, rmax)

class BaseAction(ABC):
    """Abstract base class for atom actions.

    Each subclass must implement ``execute`` and ``randomize``.
    ``randomize`` returns a dict of kwargs for ``__init__``,
    and ``apply_random`` is a convenience that randomizes + executes.
    """

    def __init__(self, atoms: Atoms):
        self.atoms = atoms

    def change_atoms(self, atoms: Atoms):
        """Change the atoms object for this action."""
        self.atoms = atoms

    @abstractmethod
    def execute(self):
        """Execute the action and return the modified Atoms object."""
        pass

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        """Sample random parameters for this action.

        Returns:
            dict of kwargs for ``__init__``, or None if no valid params found.
        """
        raise NotImplementedError(f'{cls.__name__} must implement randomize()')

    @classmethod
    def apply_random(cls, atoms, rng=None, config=None, copy=True):
        """Create a random action instance and execute it.

        Returns:
            (action, result_atoms) or (None, None) if no valid params found.
        """
        rng = rng if rng is not None else np.random.default_rng()
        target = atoms.copy() if copy else atoms
        params = cls.randomize(target, rng=rng, config=config)
        if params is None:
            return (None, None)
        action = cls(**params)
        result = action.execute()
        return (action, result)

    def __str__(self):
        return f'{self.__class__.__name__} action on {len(self.atoms)} atoms.'

class InsertBetweenAtomsAction(BaseAction):

    def __init__(self, atoms: Atoms, index1: int, index2: int, symbol: str, distance_ratio: float):
        super().__init__(atoms)
        self.index1 = index1
        self.index2 = index2
        self.symbol = symbol
        self.distance_ratio = distance_ratio
        self.distance = 0
        self.index1 = self.index1 % len(self.atoms)
        self.index2 = self.index2 % len(self.atoms)

    def execute(self):
        if 0 <= self.index1 < len(self.atoms) and 0 <= self.index2 < len(self.atoms):
            relative_position = self.atoms[self.index2].position - self.atoms[self.index1].position
            self.distance = np.linalg.norm(relative_position) * self.distance_ratio
            position = self.atoms[self.index1].position + relative_position * self.distance_ratio
            new_atom = Atoms(symbols=self.symbol, positions=[position])
            self.atoms += new_atom
            self.atoms.set_pbc(self.atoms.get_pbc())
            return self.atoms
        else:
            raise IndexError('Index out of bounds for inserting atom between two atoms.')

    def __str__(self):
        return f'Insert a {self.symbol} atom in the line between atoms at indices {self.index1} and {self.index2}, and the inserted atom must be {self.distance:.2f} angstrom from atom at {self.index1} in the cif file.'

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) < 2:
            raise ValueError('Not enough atoms to insert between')
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        idx1, idx2 = rng.choice(len(atoms), size=2, replace=False)
        symbol = str(rng.choice(cfg['symbol_pool']))
        dr = float(rng.uniform(cfg['distance_ratio_min'], cfg['distance_ratio_max']))
        dr = round(dr, cfg['decimal_places'])
        return {'atoms': atoms, 'index1': int(idx1), 'index2': int(idx2), 'symbol': symbol, 'distance_ratio': dr}

class MoveTowardsAtomAction(BaseAction):

    def __init__(self, atoms: Atoms, index1: int, index2: int, distance: float):
        super().__init__(atoms)
        self.index1 = index1
        self.index2 = index2
        self.distance = distance

    def execute(self):
        if 0 <= self.index1 < len(self.atoms) and 0 <= self.index2 < len(self.atoms):
            direction = self.atoms[self.index2].position - self.atoms[self.index1].position
            norm = np.linalg.norm(direction)
            if norm == 0:
                raise ValueError('Cannot move towards an atom at the same position.')
            direction /= norm
            move_vector = direction * self.distance
            self.atoms[self.index1].position += move_vector
            return self.atoms
        else:
            raise IndexError('Index out of bounds for moving towards another atom.')

    def __str__(self):
        return f'Move the atom at index {self.index1} towards the atom at index {self.index2} by {self.distance} angstrom in the cif file.'

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) < 2:
            raise ValueError('Not enough atoms for MoveTowardsAtomAction')
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        idx1, idx2 = rng.choice(len(atoms), size=2, replace=False)
        distance = float(rng.uniform(cfg['distance_min'], cfg['distance_max']))
        distance = round(distance, cfg['decimal_places'])
        return {'atoms': atoms, 'index1': int(idx1), 'index2': int(idx2), 'distance': distance}

class DeleteBelowAtomAction(BaseAction):
    """Delete by stored Cartesian z, without wrapping periodic images.

    include_self=True removes the center-height layer and everything below it.
    include_self=False retains the center-height layer and everything above it.
    Same layer: abs(z_j - z_center) <= z_tolerance, in angstrom.
    """
    DEFAULT_Z_TOLERANCE = 0.1

    def __init__(self, atoms: Atoms, index: int, include_self: bool=False, z_tolerance: float=DEFAULT_Z_TOLERANCE):
        super().__init__(atoms)
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise TypeError('index must be a zero-based integer')
        if not 0 <= int(index) < len(atoms):
            raise IndexError('index is outside the input structure')
        if not isinstance(include_self, (bool, np.bool_)):
            raise TypeError('include_self must be boolean')
        tolerance = float(z_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError('z_tolerance must be finite and non-negative')
        self.index = int(index)
        self.include_self = bool(include_self)
        self.z_tolerance = tolerance

    def _deletion_mask(self):
        z = self.atoms.get_positions()[:, 2]
        center_z = z[self.index]
        if self.include_self:
            return z <= center_z + self.z_tolerance
        return z < center_z - self.z_tolerance

    def execute(self):
        delete_mask = self._deletion_mask()
        deleted = int(np.count_nonzero(delete_mask))
        if deleted == 0:
            raise ValueError('No atoms satisfy the Delete Below rule')
        if deleted == len(self.atoms):
            raise ValueError('Delete Below would leave an empty structure')
        self.atoms = self.atoms[~delete_mask]
        return self.atoms

    def __str__(self):
        variant = 'Including' if self.include_self else 'Excluding'
        eps = repr(self.z_tolerance)
        return f"Delete all atoms whose z coordinate is lower than the atom at index {self.index} in the cif file. {variant} itself and atoms with the same z coordinate. Here, z refers to the Cartesian coordinate of the input structure, and atoms whose z coordinates differ from the reference atom's by at most {eps} angstrom are considered at the same height."

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) < 2:
            return None
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        tolerance = float(cfg.get('z_tolerance', cls.DEFAULT_Z_TOLERANCE))
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError('z_tolerance must be finite and non-negative')
        for _ in range(20):
            index = int(rng.integers(0, len(atoms)))
            include_self = bool(rng.random() < 0.5)
            action = cls(atoms, index, include_self, tolerance)
            deleted = int(np.count_nonzero(action._deletion_mask()))
            if 0 < deleted < len(atoms):
                return {'atoms': atoms, 'index': index, 'include_self': include_self, 'z_tolerance': tolerance}
        return None

class DeleteAroundAtomAction(BaseAction):

    def __init__(self, atoms: Atoms, index: int, radius: float):
        super().__init__(atoms)
        self.index = index
        self.radius = radius

    def execute(self):
        if 0 <= self.index < len(self.atoms):
            distances = self.atoms.get_distances(self.index, range(len(self.atoms)), mic=True)
            indices_to_delete = np.where(distances < self.radius)[0]
            if len(indices_to_delete) > 0:
                self.atoms = self.atoms[[i for i in range(len(self.atoms)) if i not in indices_to_delete]]
                return self.atoms
            else:
                raise ValueError('No atoms found within the specified radius to delete.')
        else:
            raise IndexError('Index out of bounds for deleting around an atom.')

    def __str__(self):
        return f'Delete all atoms within {self.radius} angstrom around the atom at index {self.index} in the cif file.'

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) == 0:
            raise ValueError('No atoms')
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        index = int(rng.integers(0, len(atoms)))
        rmin, rmax = _safe_radius(atoms, index, cfg['radius_min'], cfg['radius_max'])
        radius = float(rng.uniform(rmin, rmax))
        radius = round(radius, cfg['decimal_places'])
        return {'atoms': atoms, 'index': index, 'radius': radius}

class RotateAroundAtomAction(BaseAction):

    def __init__(self, atoms: Atoms, index: int, radius: float, angle: float, axis: np.ndarray):
        super().__init__(atoms)
        self.index = index
        self.radius = radius
        if not 0 <= angle < 360:
            raise ValueError('Angle must be in the range [0, 2π).')
        if np.linalg.norm(axis) == 0:
            raise ValueError('Axis of rotation cannot be a zero vector.')
        self.angle = angle
        self.axis = axis / np.linalg.norm(axis)

    def execute(self):
        if 0 <= self.index < len(self.atoms):
            center_position = self.atoms[self.index].position
            distances = self.atoms.get_distances(self.index, range(len(self.atoms)), mic=True)
            indices_to_rotate = np.where(distances < self.radius)[0]
            indices_to_rotate = [i for i in indices_to_rotate if i != self.index]
            if not indices_to_rotate:
                return self.atoms
            sub_atoms = self.atoms[indices_to_rotate]
            sub_atoms.rotate(self.angle, self.axis, center=center_position)
            for idx, sub_atom in zip(indices_to_rotate, sub_atoms):
                self.atoms[idx].position = sub_atom.position
            return self.atoms
        else:
            raise IndexError('Index out of bounds for rotating around an atom.')

    def __str__(self):
        return f'Rotate all surrounding atoms within {self.radius} angstrom of the center atom at index {self.index} by {self.angle} degree around the axis {self.axis} in the cif file. The rotation should following the right-hand rule.'

    @classmethod
    def randomize(cls, atoms, rng=None, config=None):
        if len(atoms) == 0:
            raise ValueError('No atoms')
        rng = rng if rng is not None else np.random.default_rng()
        cfg = _get_config(config)
        index = int(rng.integers(0, len(atoms)))
        rmin, rmax = _safe_radius(atoms, index, cfg['radius_min'], cfg['radius_max'])
        radius = float(rng.uniform(rmin, rmax))
        radius = round(radius, cfg['decimal_places'])
        angle = float(rng.uniform(cfg['angle_min'], cfg['angle_max']))
        angle = round(angle, cfg['decimal_places'])
        axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        axis = axes[int(rng.integers(0, len(axes)))]
        return {'atoms': atoms, 'index': index, 'radius': radius, 'angle': angle, 'axis': axis}
