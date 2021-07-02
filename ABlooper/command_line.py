import argparse
from ABlooper.ABlooper import CDR_Predictor

parser = argparse.ArgumentParser()
parser.add_argument("file_path", help="Path to the IMGT numbered antibody pdb file for which the CDRs are to be "
                                      "remodelled")
parser.add_argument("-o", "--output", help="Path to where the output model should be saved. Defaults to the same "
                                           "directory as input file.", default=None)
parser.add_argument("-H", "--heavy_chain", help="Heavy chain ID for input file. (Default is H)", default="H")
parser.add_argument("-L", "--light_chain", help="Light chain ID for input file. (Default is L)", default="L")
parser.add_argument("--confidence_score", help="Print confidence score for each loop", default="False",
                    action="store_true")

args = parser.parse_args()


def main():
    predictor = CDR_Predictor(args.file_path, chains=(args.heavy_chain, args.light_chain))
    output_file = args.output
    if output_file is None:
        output_file = args.file_path[:-4] + "_remodelled_CDRs.pdb"
    predictor.write_predictions_in_pdb_format(output_file)
    if args.confidence_score is True:
        print(predictor.decoy_diversity)
