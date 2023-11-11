import pandas as pd
import numpy as np
import requests
import time

file = pd.read_csv('data/dosage.csv', delimiter=";", usecols=['DRUG_NAME'])


def get_smiles(drug_name: str) -> str:
    """
    Sends GET requests to PubChem's servers to get the SMILES encoding for a given drug name.
    :param drug_name: A string with the drug name
    :return: A string with SMILES encoding
    """

    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/CanonicalSMILES/TXT'
    smiles = requests.get(url).text

    return smiles

