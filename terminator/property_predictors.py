"""Factory of property predictors based on strings"""
from rdkit import Chem
from modlamp.descriptors import GlobalDescriptor
from rdkit.Chem.QED import qed


def predict_qed(smiles: str) -> float:
    try:
        q = qed(Chem.MolFromSmiles(smiles, sanitize=False))
        return q, {"qed": q}
    except Exception:
        return -1, {"qed": -1}


def boman_index(sequence: str) -> float:
    """Calculate the Boman index of a protein.
    The Boman index is a measure of protein interactions (potential to bind to
    membranes or others proteins). It's the average solubility for all residues
    in the sequence. Above 2.48 is considered high binding potential.

    For details see:
        Boman, H. G. "Antibacterial peptides: basic facts and emerging concepts."
        Journal of internal medicine 254.3 (2003): 197-215.

    Args:
        sequence (str): An AA sequence

    Returns:
        float: The boman index.
    """
    try:
        sequence = sequence.strip().upper()
        desc = GlobalDescriptor(sequence)
        desc.boman_index()
        b = float(desc.descriptor)
        return b, {"boman": b}
    except Exception:
        return -100, {"boman": -100}


PREDICT_FACTORY = {"qed": predict_qed, "boman": boman_index}
