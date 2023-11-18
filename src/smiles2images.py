from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd


def chemcepterize_mol(mol: Chem.Mol, embed: float = 20.0, res: float = 0.5) -> np.ndarray:
    """
    Code from Esben Jannik Bjerrum. Published on Cheminformania on 28.11.2017.
    Convert a SMILES string into an image the molecular structure (80x80 pixels)
    :param str mol: SMILES string of a chemical compound (drug)
    :param float embed:  embedding size
    :param float res: image resolution
    :return np.ndarray vect:
    """
    if mol is None:
        return None

    dims = int(embed * 2 / res)
    cmol = Chem.Mol(mol.ToBinary())
    cmol.ComputeGasteigerCharges()
    AllChem.Compute2DCoords(cmol)
    coords = cmol.GetConformer(0).GetPositions()
    vect = np.zeros((dims, dims, 4))
    # Bonds first
    for i, bond in enumerate(mol.GetBonds()):
        bondorder = bond.GetBondTypeAsDouble()
        bidx = bond.GetBeginAtomIdx()
        eidx = bond.GetEndAtomIdx()
        bcoords = coords[bidx]
        ecoords = coords[eidx]
        frac = np.linspace(0, 1, int(1 / res * 2))  #
        for f in frac:
            c = (f * bcoords + (1 - f) * ecoords)
            idx = int(round((c[0] + embed) / res))
            idy = int(round((c[1] + embed) / res))
            # Save in the vector first channel
            vect[idx, idy, 0] = bondorder
    # Atom Layers
    for i, atom in enumerate(cmol.GetAtoms()):
        idx = int(round((coords[i][0] + embed) / res))
        idy = int(round((coords[i][1] + embed) / res))
        # Atomic number
        vect[idx, idy, 1] = atom.GetAtomicNum()
        # Gasteiger Charges
        charge = atom.GetProp("_GasteigerCharge")
        vect[idx, idy, 3] = charge
        # Hybridization
        hyptype = atom.GetHybridization().real
        vect[idx, idy, 2] = hyptype

    return vect


if __name__ == '__main__':
    filepath = '../data/smiles.csv'

    file = pd.read_csv(filepath, delimiter=',', header=0)

    # Convert SMILES strings into RDKit molecular structures
    file['mol'] = file['smiles'].apply(lambda x: Chem.MolFromSmiles(x) if pd.isna(x) == False else None)
    file['chem'] = file['mol'].apply(chemcepterize_mol)

    # outpath = filepath
    # file.to_csv(outpath, index=False)
