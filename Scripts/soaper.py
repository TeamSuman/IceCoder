# Standard library
# MDAnalysis library and utilities
# import MDAnalysis as mda  # type: ignore
import numpy as np

# SOAP library
from ase import Atoms  # type: ignore
from dscribe.descriptors import SOAP  # type: ignore
from MDAnalysis.lib.mdamath import triclinic_vectors as tv  # type: ignore


class soapFromUniverse:
    def __init__(self, n_max=8, l_max=6, sigma=0.25, r_cut=10.0):

        # Setting up the SOAP descriptor
        self.soap = SOAP(
            species=[1, 8], periodic=True, r_cut=r_cut, n_max=n_max, l_max=l_max, average="off", rbf="gto", sigma=sigma
        )

    # Define the system under study
    def soaper(self, u, frame=0):
        u.trajectory[frame]

        # Select all potential water/ice atoms/residues
        water_atoms = u.select_atoms(
            "resname ICE or resname SOL or resname WAT or resname HOH or resname MW or resname mW"
        )

        # Determine if we are dealing with a monatomic water model (like mW)
        is_monatomic = False
        if len(water_atoms) > 0:
            has_hydrogen = any(
                (hasattr(atom, "element") and atom.element in ("H", "H_")) or "H" in atom.name or "h" in atom.name
                for atom in water_atoms
            )
            is_monatomic = not has_hydrogen or (len(water_atoms) == water_atoms.n_residues)

        # Determine cell dimensions safely
        dim = u.dimensions
        if dim is None or np.ndim(dim) == 0 or len(dim) < 6 or np.all(dim == 0):
            dim = u.trajectory[0].dimensions
            if dim is None or np.ndim(dim) == 0 or len(dim) < 6 or np.all(dim == 0):
                raise ValueError(
                    f"Invalid box/unit-cell dimensions: {dim!r}. "
                    "Expected at least 6 non-zero values, e.g. "
                    "[lx, ly, lz, alpha, beta, gamma]."
                )
                # dim = np.array([30.0, 30.0, 30.0, 90.0, 90.0, 90.0])
        cell_vectors = tv(dim)

        if is_monatomic:
            # Monatomic water model (e.g. mW) where each atom represents one water molecule
            pos = water_atoms.positions
            conv = Atoms(cell=cell_vectors, positions=pos, symbols=["O"] * len(pos), pbc=True)
            centers = np.arange(len(pos))
        else:
            # Standard multi-atom water model (e.g. SPC/E, TIP3P, TIP4P)
            pos = u.select_atoms("(resname ICE or resname SOL or resname WAT or resname HOH) and not name MW").positions
            ow_atoms = u.select_atoms(
                "(resname ICE or resname SOL or resname WAT or resname HOH) and (name OW or name O or name OH2)"
            )
            conv = Atoms(cell=cell_vectors, positions=pos, symbols=["O", "H", "H"] * len(ow_atoms), pbc=True)
            centers = np.arange(0, len(pos), 3)

        return self.soap.create(conv, centers=centers)
