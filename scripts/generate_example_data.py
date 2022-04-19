"""
Generate example data starting from a .smi file.

We use QED of molecules as an example.
"""

import argparse
import os

from rdkit import Chem
from rdkit.Chem import QED

parser = argparse.ArgumentParser()
parser.add_argument("input_filepath", type=str, help="path to the .smi file.")
parser.add_argument("output_filepath", type=str, help="output where to store the data.")


def main() -> None:
    """Generate example data."""
    args = parser.parse_args()
    input_filepath = args.input_filepath
    output_filepath = args.output_filepath

    with open(input_filepath, "rt") as fpr:
        with open(output_filepath, "wt") as fpw:
            smiles_generator = (line.strip().split("\t")[0] for line in fpr)
            for smiles in smiles_generator:
                try:
                    fpw.write(
                        f"<qed>{QED.qed(Chem.MolFromSmiles(smiles)):.4}|{smiles}{os.linesep}"
                    )
                except Exception:
                    print(f"Problem processing SMILES={smiles}")


if __name__ == "__main__":
    main()
