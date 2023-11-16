import pandas as pd
import numpy as np
import requests
import time
from tqdm import tqdm

from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles


def get_smiles(drug_name: str) -> str | None:
    """
    Get SMILES encoding from PubChem for a given drug name
    (!!! MAX. 5 REQUESTS PER SECOND)
    :param str drug_name: A string with the drug name
    :return str|None smiles: A string with SMILES encoding
    """

    # SMILES encodings can be found in this URL in PubChem's website
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/CanonicalSMILES/TXT'
    # Send a GET request to the server and get the SMILES
    pubchem_response = requests.get(url).text

    if 'Status: 404' in pubchem_response:
        return None

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
    drug_col = 'DRUG_NAME'

    # Path to data source
    file = pd.read_csv('../data/unprocessed/dosage.csv',
                       delimiter=";",
                       usecols=[drug_col]
                       )

    smiles_mapping = {}  # Keys = drug names, values = corresponding SMILES embedding
    unique_drugs = file[drug_col].unique()  # List of unique drugs in the given dataset
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
