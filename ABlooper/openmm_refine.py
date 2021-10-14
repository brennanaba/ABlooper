import pdbfixer
import tempfile
from io import StringIO
from simtk.openmm import app, LangevinIntegrator, CustomExternalForce
from simtk import unit


ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms
spring_unit = ENERGY / (LENGTH ** 2)


def openmm_refine(pdb_txt, CDR_definitions, spring_constant=10):
    with tempfile.NamedTemporaryFile("wt") as tmp:
        tmp.writelines(pdb_txt)
        fixer = pdbfixer.PDBFixer(tmp.name)

    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # Using amber14 recommended protein force field
    forcefield = app.ForceField("amber14/protein.ff14SB.xml")

    # Fill in the gaps with OpenMM Modeller
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(forcefield)

    # Set up force field
    system = forcefield.createSystem(modeller.topology)

    # Only move the CDRs
    movable = []
    for chain, rang in CDR_definitions.values():
        for res in range(rang[0], rang[1] + 1):
            movable.append((chain, res))

    # Keep atoms close to initial prediction
    force = CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", spring_constant*spring_unit)
    for p in ["x0", "y0", "z0"]:
        force.addPerParticleParameter(p)

    for residue in modeller.topology.residues():
        if (residue.chain.id, int(residue.id)) not in movable:
            for atom in residue.atoms():
                system.setParticleMass(atom.index, 0.0)
        else:
            for atom in residue.atoms():
                if atom.name in ["CA", "CB", "N", "C"]:
                    force.addParticle(atom.index, modeller.positions[atom.index])
    system.addForce(force)

    # Set up integrator
    integrator = LangevinIntegrator(100, 0.01, 0.0)

    # Set up the simulation
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    # Minimize the energy
    simulation.minimizeEnergy()

    out_handle = StringIO()
    app.PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(),
                          out_handle, keepIds=True)
    out_txt = out_handle.getvalue()
    out_handle.close()

    return [x+"\n" for x in out_txt.split("\n") if x[:3] in ["ATO", "TER"]]
