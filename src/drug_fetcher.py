import os
import time
import pandas as pd
import requests
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm
import numpy as np


def get_smiles(drug_name: str) -> str | None:
    """
    Get SMILES encoding from PubChem for a given drug name
    (!!! MAX. 5 REQUESTS PER SECOND)
    :param str drug_name: A string with the drug name
    :return str|None smiles: A string with SMILES encoding
    """

    # SMILES encodings can be found in this URL in PubChem's website
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/CanonicalSMILES/json'
    # Send a GET request to the server and get the SMILES
    pubchem_response = requests.get(url)

    if pubchem_response.status_code == 404:
        return None

    pubchem_response = pubchem_response.json()['PropertyTable']['Properties'][0]['CanonicalSMILES']

    # Remove newlines and spaces from SMILES string
    pubchem_response = pubchem_response.replace('\n', '')
    pubchem_response = pubchem_response.replace(' ', '')

    return pubchem_response


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


if __name__ == '__main__':

    # Name of column where drug names can be found
    drug_col = 'drug'


    # Path to data source
    train_df = pd.read_csv('../data/train.csv',
                           delimiter=",",
                           usecols=[drug_col]
                           )

    test_df = pd.read_csv('../data/test.csv',
                          delimiter=",",
                          usecols=[drug_col]
                          )

    print(train_df.columns)

    drugs = np.array([train_df.drug.unique(), test_df.drug.unique()])
    drugs = drugs.flatten()
    unique_drugs = list(set(drugs))

    smiles_mapping = {}  # Keys = drug names, values = corresponding SMILES embedding
    no_unique_drugs = len(unique_drugs)  # Number of unique drugs

    for i, drug in enumerate(tqdm(unique_drugs)):
        response = get_smiles(drug)
        smiles_mapping[drug] = response

        # Sleep 1 second every 5 requests (per PubChem's request)
        if i % 5 == 0:
            time.sleep(1)

    not_found_count = list(smiles_mapping.values()).count(None)

    print(f"{not_found_count} out of {no_unique_drugs} drugs were not found on PubChem!\n"
          f"Please check the spelling of the drug name in the dataset.")

    smiles_df = pd.DataFrame(list(smiles_mapping.items()), columns=['drug', 'smiles'])

    smiles_df.to_csv('../data/smiles.csv', index=False)
