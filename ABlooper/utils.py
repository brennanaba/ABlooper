import torch
import numpy as np
import copy
from einops import rearrange
import sys
import os

aa1 = "ACDEFGHIKLMNPQRSTVWY"
aa3 = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER",
       "THR", "VAL", "TRP", "TYR", ]

short2long = {}
long2short = {}
short2num = {}

for ind in range(0, 20):
    long2short[aa3[ind]] = aa1[ind]
    short2long[aa1[ind]] = aa3[ind]
    short2num[aa1[ind]] = ind


def encode(x, classes):
    """ One hot encodes a scalar x into a vector of length classes.
    This is the function used for Sequence encoding.
    """
    one_hot = np.zeros(classes)
    one_hot[x] = 1

    return one_hot


def one_hot(num_list, classes=20):
    """ One hot encodes a 1D vector x.
    This is the function used for Sequence encoding.
    """
    end_shape = (len(num_list), classes)
    finish = np.zeros(end_shape)
    for i in range(end_shape[0]):
        finish[i] = encode(num_list[i], classes)

    return finish


def filt(x, chain, loop_range):
    """ Function to select residues in a certain chain within a given range.

    If the pdb line contains an atom belonging to the desired chain within the range it returns True.
    """
    if x[:4] == "ATOM" and x[21] == chain:
        if loop_range[0] <= int(x[22:26]) <= loop_range[1]:
            return True

    return False


def positional_encoding(sequence, n=5):
    """ Gives the network information on how close each resdiue is to the anchors
    """
    encs = []
    L = len(sequence)
    for i in range(n):
        encs.append(np.cos((2 ** i) * np.pi * np.arange(L) / L))
        encs.append(np.sin((2 ** i) * np.pi * np.arange(L) / L))

    return np.array(encs).transpose()


def res_to_atom(amino, n_atoms=4):
    """ Adds a one-hot encoded vector to each node describing what atom type it is.

    It also reshapes the input tensor.
    """
    residue_feat = rearrange(amino, "i d -> i () d")
    atom_type = rearrange(torch.eye(n_atoms, device=amino.device), "a d -> () a d")

    i = residue_feat.shape[0]
    atom_feat = torch.cat([residue_feat.repeat(1, n_atoms, 1), atom_type.repeat(i, 1, 1)], dim=-1)

    return atom_feat


def which_loop(loop_seq, cdr):
    """ Adds a one-hot encoded vector to each node describing which CDR it belongs to.
    """
    CDRs = ["H1", "H2", "H3", "L1", "L2", "L3", "Anchor"]
    loop = np.zeros((len(loop_seq), len(CDRs)))
    loop[:, -1] = 1
    loop[2:-2] = np.array([1.0 if cdr == x else 0.0 for x in (CDRs)])[None].repeat(len(loop_seq) - 4, axis=0)

    return loop


def rmsd(loop1, loop2):
    """ Simple rmsd calculation for numpy arrays.
    """
    return np.sqrt(np.mean(((loop1 - loop2) ** 2).sum(-1)))


def to_pdb_line(atom_id, atom_type, amino_type, chain_ID, residue_id, coords):
    """Puts all the required info into a .pdb format
    """
    x, y, z = coords
    insertion = "$"
    if type(residue_id) is str:
        if residue_id[-1].isalpha():
            insertion = residue_id[-1]
            residue_id = int(residue_id[:-1])
        else:
            residue_id = int(residue_id)
    line = "ATOM  {:5d}  {:3s} {:3s} {:1s} {:3d}{:2s}  {:8.3f}{:8.3f}{:8.3f}  1.00  0.00           {}  \n"
    line = line.format(atom_id, atom_type, amino_type, chain_ID, residue_id, insertion, x, y, z, atom_type[0])

    return line.replace("$", " ")


def prepare_input_loop(CDR_coords, CDR_seq, CDR):
    """ Generates input features to be fed into the network
    """
    CDR_input_coords = copy.deepcopy(CDR_coords)
    CDR_input_coords[1:-1] = np.linspace(CDR_coords[1], CDR_coords[-2], len(CDR_coords) - 2)
    CDR_input_coords = rearrange(torch.tensor(CDR_input_coords), "i a d -> () (i a) d").float()

    one_hot_encoding = one_hot(np.array([short2num[amino] for amino in CDR_seq]))
    loop = which_loop(CDR_seq, CDR)
    positional = positional_encoding(CDR_seq)
    encoding = res_to_atom(torch.tensor(np.concatenate([one_hot_encoding, positional, loop], axis=1)).float())
    encoding = rearrange(encoding, "i a d -> () (i a) d")

    return CDR_input_coords, encoding


def stop_print(func, *args, **kwargs):
    """ Runs a function func with whatever arguments are needed while blocking all print statements
    """
    with open(os.devnull, "w") as devNull:
        original = sys.stdout
        sys.stdout = devNull
        func(*args, **kwargs)
        sys.stdout = original
