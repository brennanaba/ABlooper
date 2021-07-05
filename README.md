# ABlooper
The main area of structural variation in antibodies is concentrated in their six complementarity determining regions (CDRs), with the most variable being the H3 loop. The sequence and structure variability of H3 make it particularly challenging to model. Recent work using deep learning to improve protein structure prediction has shown promising results. In this work we present ABlooper, an end-to-end equivariant deep-learning based CDR loop structure prediction tool. ABlooper can predict the structure of CDR loops to a high level of accuracy while providing a confidence estimate for each of its predictions. On the models of the Rosetta Antibody Benchmark, ABlooper makes predictions with an average H3 RMSD of 2.46Å, which drops to 1.91Å when considering only the 70% most confident predictions.

## Install

To download:

```bash
$ git clone https://github.com/brennanaba/ABlooper.git
```

To install

```bash
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


I would recommend using the command line if you just want a quick antibody model, no questions asked. The python class will work better if you want to incorporate this into a pipeline or access other details such as confidence score or RMSD to original model. Both of which can be obtained as follows:


```python
rmsd_from_input = pred.calculate_BB_rmsd_wrt_input()
confidence_score = pred.decoy_diversity 
```



