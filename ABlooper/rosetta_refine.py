import pyrosetta
import tempfile
from ABlooper.utils import stop_print

stop_print(pyrosetta.init, "-mute all")

# The code in this file is an adaptation of scripts written by Jeffrey Ruffolo (jeffreyruffolo).
# Many thanks to him for his help


def get_loop_ranges(chain, res, info):
    """ Find loops in PyRosetta pose numbering

    """
    start = info.pdb2pose(chain, res[0])
    end = info.pdb2pose(chain, res[1])

    # Basically if you can't find the start or end residues, try the next residue until you run out.
    if start == 0:
        i = res[0]
        while (start == 0) and (i > 18):
            i -= 1
            start = info.pdb2pose(chain, i)

    if end == 0:
        i = res[1]
        while (end == 0) and (i < 122):
            i += 1
            end = info.pdb2pose(chain, i)

    return start, end


def get_fa_min_mover(loop_ranges, pose, min_iter):
    """ Create full-atom minimization mover

    """
    # Create full-atom score function with terms for fixing bad bond lengths
    sf = pyrosetta.create_score_function('ref2015_cst')
    sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded, 1)
    sf.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.pro_close, 0)

    # Try to minimize only the loops
    try:
        loops = pyrosetta.rosetta.protocols.loops.Loops()
        [loops.add_loop(r[0], r[1]) for r in loop_ranges]

        mmap = pyrosetta.rosetta.protocols.loops.move_map_from_loops(
            pose, loops, False)

    # If selecting the loops doesn't work, minimise the whole thing
    except RuntimeError:
        mmap = pyrosetta.rosetta.core.kinematics.MoveMap()
        mmap.set_bb(True)
        mmap.set_chi(True)
        mmap.set_jump(True)

    # Create MinMover acting in cartesian space
    min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover(
        mmap, sf, 'lbfgs_armijo_nonmonotone', 0.0001, True)
    min_mover.max_iter(min_iter)
    min_mover.cartesian(True)

    return min_mover


def rosetta_refine(pdb_txt, CDR_definitions, min_iter=500):
    """ PyRosetta protocol for minimization refinement of protein structure

    """
    # Load input PDB into pose
    with tempfile.NamedTemporaryFile("wt") as tmp:
        tmp.writelines(pdb_txt)
        pose = pyrosetta.pose_from_pdb(tmp.name)

    # Get PDB info
    info = pose.pdb_info()

    # Define CDR loops
    loop_ranges = [get_loop_ranges(*x, info) for x in CDR_definitions.values()]

    # Create movers
    cst_mover = pyrosetta.rosetta.protocols.relax.AtomCoordinateCstMover()
    cst_mover.cst_sidechain(False)
    min_mover = get_fa_min_mover(
        loop_ranges=loop_ranges,
        pose=pose,
        min_iter=min_iter,
    )
    idealize_mover = pyrosetta.rosetta.protocols.idealize.IdealizeMover()

    # Refine structure
    cst_mover.apply(pose)
    min_mover.apply(pose)
    idealize_mover.apply(pose)

    # Save refined structure to string
    with tempfile.NamedTemporaryFile("rt") as tmp:
        pose.dump_pdb(tmp.name)
        txt = tmp.readlines()

    # Remove Rosetta details
    return [x for x in txt if x[:3] in ["ATO", "TER"]]
