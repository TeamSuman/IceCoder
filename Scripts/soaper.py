# Standard library 
import numpy as np
# SOAP library
from ase import Atoms
from dscribe.descriptors import SOAP
# MDAnalysis library and utilities
import MDAnalysis as mda
from MDAnalysis.lib.mdamath import triclinic_vectors as tv

class soapFromUniverse:
    def __init__(self, n_max = 8, l_max = 6, sigma = 0.25, r_cut = 10.0):
        
        # Setting up the SOAP descriptor
        self.soap = SOAP(
            species=[1, 8],
            periodic=True,
            r_cut = r_cut,
            n_max = n_max,
            l_max= l_max,
            average = 'off',
            rbf='gto',
            sigma = sigma
        )


    # Define the system under study: NaCl in a conventional cell.
    def soaper(self, u, frame = 0):
        u.trajectory[frame]
        pos = u.select_atoms("resname ICE or resname SOL and not name MW").positions
        #scale_pos = pos - pos.min(axis = 0)
        #scale_pos /= scale_pos.max()
        conv = Atoms(
            cell=tv(u.dimensions),
            positions = pos,
            #positions=pos*(1/dens)**3, 
            symbols=["O", "H", "H"]*len(u.select_atoms("resname ICE or resname SOL and name OW").positions),pbc = True)
        return self.soap.create(conv, centers=np.arange(0, len(pos), 3))