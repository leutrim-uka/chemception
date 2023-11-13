import pandas as pd
import numpy as np
import requests
import time

from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles

file = pd.read_csv('data/dosage.csv', delimiter=";", usecols=['DRUG_NAME'])


def get_smiles(drug_name: str) -> str:
    """
    Get SMILES encoding from PubChem for a given drug name
    (!!! MAX. 5 REQUESTS PER SECOND)
    :param str drug_name: A string with the drug name
    :return str smiles: A string with SMILES encoding
    """

    # SMILES encodings can be found in this URL in PubChem's website
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/CanonicalSMILES/TXT'
    # Send a GET request to the server and get the SMILES
    smiles = requests.get(url).text
    # Remove newlines and spaces from SMILES string
    smiles = smiles.replace('\n', '')
    smiles = smiles.replace(' ', '')

    return smiles


def get_2d_images(drug_name: str, width: int = 80, height: int = 80) -> bytes:
    """
    Get 2D image from PubChem for a given drug name
    (!!! MAX. 5 REQUESTS PER SECOND)
    :param str drug_name: A string with the drug name
    :param int width: Desired image width
    :param int height: Desired image height
    :return bytes encoded_image: Image expressed in bytes
    """

    image_size = f"{width}x{height}"
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/PNG?image_size={image_size}"
    response = requests.get(url)
    encoded_image = response.content

    return encoded_image


def smiles2image(smiles: str, width: int, height: int):
    """

    :param str smiles:
    :param int width:
    :param int height:
    :return:
    """
    # Remove newlines and spaces from SMILES string
    smiles = smiles.replace(' ', '')
    smiles = smiles.replace('\n', '')

    mol = MolFromSmiles(smiles)

    pass



