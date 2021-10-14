import torch
import numpy as np
from einops import rearrange
from ABlooper.models import DecoyGen
from ABlooper.utils import filt, rmsd, to_pdb_line, short2long, long2short, prepare_input_loop
import os

try:
    from ABlooper.rosetta_refine import rosetta_refine
except ModuleNotFoundError:
    rosetta_available = False
else:
    rosetta_available = True

try:
    from ABlooper.openmm_refine import openmm_refine
except ModuleNotFoundError:
    openmm_available = False
else:
    openmm_available = True

device = "cuda" if torch.cuda.is_available() else "cpu"
current_directory = os.path.dirname(os.path.realpath(__file__))

default_model = DecoyGen().float().to(device)
path = os.path.join(current_directory, "trained_model", "final_model")
default_model.load_state_dict(torch.load(path, map_location=torch.device(device)))


class CDR_Predictor:
    def __init__(self, pdb_file, chains=("H", "L"), model=None, refine=False, refine_method="openmm"):
        """ Class used to handle remodelling of CDRs.

        :param pdb_file: File of IMGT numbered antibody structure in .pdb format. Must include heavy and light chain
        :param chains: Name of heavy and light chain to be remodelled in the file
        :param model: Trained neural network. If left as None a pretrained network is used.
        """
        self.pdb_file = pdb_file
        self.chains = chains
        self.model = model if model is not None else default_model
        self.CDR_with_anchor_slices = {
            "H1": (chains[0], (25, 39)),
            "H2": (chains[0], (55, 66)),
            "H3": (chains[0], (105, 119)),
            "L1": (chains[1], (22, 42)),
            "L2": (chains[1], (54, 71)),
            "L3": (chains[1], (103, 119))}
        self.__atoms = ["CA", "N", "C", "CB"]

        self.predicted_CDRs = {}
        self.all_decoys = {}
        self.decoy_diversity = {}

        with open(self.pdb_file) as file:
            self.pdb_text = [line for line in file.readlines()]

        # For all three of these I extract the loop plus two anchors at either side as these are needed for the model.
        self.CDR_text = {CDR: [x for x in self.pdb_text if filt(x, *self.CDR_with_anchor_slices[CDR])] for CDR in
                         self.CDR_with_anchor_slices}
        self.CDR_sequences = {
            CDR: "".join([long2short[x.split()[3][-3:]] for x in self.CDR_text[CDR] if x.split()[2] == "CA"]) for CDR in
            self.CDR_with_anchor_slices}
        self.__extract_BB_coords()

        # Here I don't extract the anchors as this is only needed for writing predictions to pdb file.
        self.CDR_numberings = {CDR: [x.split()[5] for x in self.CDR_text[CDR] if x.split()[2] == "CA"][2:-2] for CDR in
                               self.CDR_text}
        self.CDR_start_atom_id = {CDR: int([x.split()[1] for x in self.CDR_text[CDR] if x.split()[2] == "N"][2]) for CDR
                                  in self.CDR_text}

        # If rosetta or openmm are not available refinement is not possible
        if (not rosetta_available) and (not openmm_available) and refine:
            print("Neither PyRosetta nor OpenMM are available. Refinement is not possible")
        self.refine = refine and (rosetta_available or openmm_available)
        self.refine_method = refine_method

    def __extract_BB_coords(self):
        self.CDR_BB_coords = {}

        for CDR in self.CDR_with_anchor_slices:
            loop = self.CDR_text[CDR]

            coors = np.zeros((len(self.CDR_sequences[CDR]), 4, 3))
            coors[...] = float("Nan")

            i = 0
            res = loop[i].split()[5]

            for line in loop:
                cut = line.split()
                if cut[5] != res:
                    res = cut[5]
                    i += 1
                if cut[2] in self.__atoms:
                    j = self.__atoms.index(cut[2])
                    # Using split for coords doesn't always work. Following Biopython approach:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coors[i, j] = np.array([x, y, z])

            # If missed CB (GLY) then add CA instead
            coors[:, 3] = np.where(np.all(coors[:, 3] != coors[:, 3], axis=-1, keepdims=True), coors[:, 0], coors[:, 3])
            self.CDR_BB_coords[CDR] = coors

    def __prepare_model_input(self):
        encodings = []
        geomins = []

        for CDR in self.CDR_BB_coords:
            geom, encode = prepare_input_loop(self.CDR_BB_coords[CDR], self.CDR_sequences[CDR], CDR)
            encodings.append(encode)
            geomins.append(geom)

        return torch.cat(encodings, axis=1).detach(), torch.cat(geomins, axis=1).detach()

    def predict_CDRs(self):

        with torch.no_grad():
            input_encoding, input_geometry = self.__prepare_model_input()
            output = self.model(input_encoding.to(device), input_geometry.to(device))

            for i, CDR in enumerate(self.CDR_with_anchor_slices):
                output_CDR = output[:, input_encoding[0, :, 30 + i] == 1.0]
                self.all_decoys[CDR] = rearrange(output_CDR, "b (i a) d -> b i a d", a=4).cpu().numpy()
                self.predicted_CDRs[CDR] = rearrange(output_CDR.mean(0), "(i a) d -> i a d", a=4).cpu().numpy()
                self.decoy_diversity[CDR] = (output_CDR[None] - output_CDR[:, None]).pow(2).sum(-1).mean(-1).pow(
                    1 / 2).sum().item() / 20

        return self.predicted_CDRs

    def calculate_BB_rmsd_wrt_input(self):
        assert "H1" in self.predicted_CDRs, "Predict it before comparing"
        rmsds = {}

        for CDR in self.CDR_BB_coords:
            rmsds[CDR] = rmsd(self.CDR_BB_coords[CDR][2:-2], self.predicted_CDRs[CDR])

        return rmsds

    def __convert_predictions_into_text_for_each_CDR(self):
        pdb_format = {}

        pdb_atoms = ["N", "CA", "C", "CB"]
        permutation_to_reorder_atoms = [1, 0, 2, 3]

        for CDR in self.CDR_start_atom_id:
            new_text = []
            BB_coords = self.predicted_CDRs[CDR]
            seq = self.CDR_sequences[CDR][2:-2]
            numbering = self.CDR_numberings[CDR]
            atom_id = self.CDR_start_atom_id[CDR]
            chain = self.CDR_with_anchor_slices[CDR][0]

            for i, amino in enumerate(BB_coords):
                amino_type = short2long[seq[i]]
                for j, coord in enumerate(amino[permutation_to_reorder_atoms]):
                    if (pdb_atoms[j] == "CB") and (amino_type == "GLY"):
                        continue
                    new_text.append(to_pdb_line(atom_id, pdb_atoms[j], amino_type, chain, numbering[i], coord))
                    atom_id += 1

            pdb_format[CDR] = new_text
        return pdb_format

    def write_predictions_in_pdb_format(self, file_name=None, to_be_rewritten=None):
        if to_be_rewritten is None:
            # This should probably be left as is to avoid clashes between loops
            to_be_rewritten = ["H1", "H2", "H3", "L1", "L2", "L3"]
        if "H1" not in self.predicted_CDRs:
            self.predict_CDRs()

        text_prediction_per_CDR = self.__convert_predictions_into_text_for_each_CDR()
        old_text = self.pdb_text

        for CDR in to_be_rewritten:
            new = True
            new_text = []
            chain, CDR_slice = self.CDR_with_anchor_slices[CDR]
            CDR_slice = (CDR_slice[0] + 2, CDR_slice[1] - 2)

            for line in old_text:
                if not filt(line, chain, CDR_slice):
                    new_text.append(line)
                elif new:
                    new_text += text_prediction_per_CDR[CDR]
                    new = False
                else:
                    continue
            old_text = new_text

        header = [
            "REMARK    CDR LOOPS REMODELLED USING ABLOOPER                                   \n"]

        if self.refine:
            if self.refine_method == "openmm" and openmm_available:
                old_text = openmm_refine(old_text, self.CDR_with_anchor_slices)
                header.append("REMARK    REFINEMENT DONE USING OPENMM" + 42 * " " + "\n")
            else:
                old_text = rosetta_refine(old_text, self.CDR_with_anchor_slices)
                header.append("REMARK    REFINEMENT DONE USING PYROSETTA" + 39 * " " + "\n")

        new_text = "".join(header + old_text)

        if file_name is None:
            return "".join(new_text)
        else:
            with open(file_name, "w+") as file:
                file.write(new_text)

    def __repr__(self):
        return "CDR_Predictor: {}".format(self.pdb_file.split("/")[-1])
