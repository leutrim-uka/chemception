# Chemception
This repository is a supplement of the paper on Chemception's applications in predicting cancer cells' response to
different drugs (compounds).

## Background & Idea
Recent machine learning research in the domain of oncology utilizes SMILES encoding to represent the molecular structure
of the drugs used to treat cancer. They improve upon baselines built with Morgan's fingerprints. In turn, this paper uses 
SMILES to build a baseline, while exploring the abilities of Chemception to further improve on the topic by using image 
representations of the molecular structures of compounds, instead of a textual representation via SMILES.

# Implementation
## SMILES embeddings
<img src="https://github.com/leutrim-uka/Chemception/assets/67911249/efd5c06a-dd8e-4c82-b6bf-b0bec35b69b4" >


## Squeeze & Unsqueeze as wrapper classes
The Squeeze and Unsqueeze classes are essentially wrappers around the torch.squeeze and torch.unsqueeze functions, respectively. They allow these functions to be used as layers within a nn.Sequential model. The difference between using these classes and directly using torch.squeeze and torch.unsqueeze is that the latter are just functions, while the former are PyTorch modules. This means that when you use the Squeeze and Unsqueeze classes, you can include these operations as part of a PyTorch model and take advantage of features like automatic differentiation and GPU acceleration.

In other words, if you’re building a model using nn.Sequential, you would need to use these wrapper classes (or similar) to include squeeze and unsqueeze operations in your model. If you’re just writing a script, and you need to squeeze or unsqueeze a tensor, you can use torch.squeeze and torch.unsqueeze directly.

It’s also worth noting that the Squeeze class will by default remove all dimensions of size 1 from the tensor, while the Unsqueeze class will add a dimension of size 1 at the specified position.