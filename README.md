# ABlooper
Antibodies are a key component of the immune system and have been extensively used as biotherapeutics. Accurate knowledge of their structure is central to understanding their function. The key area for antigen binding and the main area of structural variation in antibodies are concentrated in their six complementarity determining regions (CDRs), with the most variable being the CDR-H3 loop. The sequence and structure variability of CDR-H3 make it particularly challenging to model. Recently, deep learning methods have offered a step change in our ability to predict protein structures. In this work we present ABlooper, an end-to-end equivariant deep-learning based CDR loop structure prediction tool. ABlooper predicts the structure of CDR loops with high accuracy and provides a confidence estimate for each of its predictions. On the models of the Rosetta Antibody Benchmark, ABlooper makes predictions with an average H3 RMSD of 2.45Å, which drops to 2.02Å when considering only its 76\% most confident predictions. 

## Install

To install via PyPi

```bash
$ pip install ABlooper
```

To download and install the latest version from github:

```bash
$ git clone https://github.com/brennanaba/ABlooper.git
$ pip install ABlooper/
```

This package requires PyTorch. If you do not already have PyTorch installed, you can do so following these <a href="https://pytorch.org/get-started/locally/">instructions</a>.

## Usage

To use ABlooper, you will need an IMGT numbered antibody model. If you do not already have an antibody model, you can generate one using <a href="http://opig.stats.ox.ac.uk/webapps/newsabdab/sabpred/abodybuilder/">ABodyBuilder</a>.

To remodel the CDRs of an existing antibody model using the command line:

```bash
$ ABlooper my_antibody_model.pdb --output ABlooper_model.pdb --heavy_chain H --light_chain L
```

To remodel the CDRs of an existing model using the python API:

```python
from ABlooper import CDR_Predictor

input_path = "my_antibody_model.pdb"
output_path = "ABlooper_model.pdb"

pred = CDR_Predictor(input_path, chains = ("H", "L"))
pred.write_predictions_in_pdb_format(output_path)
```


I would recommend using the command line if you just want a quick antibody model. If speed is a priority, it is probably best to just use the trained pytorch module, as reading and writting a pdb file is slower that running the model. The python class will work best if you want to incorporate CDR prediction into a pipeline or access other details such as confidence score or RMSD to original model. Both of which can be obtained as follows:


```python
rmsd_from_input = pred.calculate_BB_rmsd_wrt_input()
confidence_score = pred.decoy_diversity 
```

## Citing this work

The code and data in this package is based on the following paper <a href="https://www.biorxiv.org/content/10.1101/2021.07.26.453747v1">ABlooper</a>. If you use it, please cite:

```tex
@article {ABLooper,
	author = {Abanades, Brennan and Georges, Guy and Bujotzek, Alexander and Deane, Charlotte M},
	title = {ABlooper: Fast accurate antibody CDR loop structure prediction with accuracy estimation},
	year = {2021},
	doi = {10.1101/2021.07.26.453747},
	URL = {https://www.biorxiv.org/content/early/2021/07/26/2021.07.26.453747},
	journal = {bioRxiv}
}
```

