# -*- coding: utf-8 -*-
import os
import sys
from openmm import *
from openmm.app import *
from openmm.unit import *
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.toolkit.topology import Molecule
import parmed as pmd


class MDSimulator:
    def __init__(self, work_dir):
        # Accept absolute path or subdirectory name under working_directory/
        if os.path.isabs(work_dir):
            self.work_dir = work_dir
        else:
            self.work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working_directory", work_dir)
        self.results_dir = os.path.join(self.work_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"Work directory: {self.work_dir}")
        print(f"Results directory: {self.results_dir}")

    def load_system(self, pdb_name, sdf_name=None, box_padding=1.0 * nanometers):
        pdb_path = os.path.join(self.work_dir, pdb_name)
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")

        pdb = PDBFile(pdb_path)
        modeller = Modeller(pdb.topology, pdb.positions)

        if sdf_name:
            mol_path = os.path.join(self.work_dir, sdf_name)
            if not os.path.exists(mol_path):
                raise FileNotFoundError(f"SDF file not found: {mol_path}")

            print(f"Loading ligand from: {mol_path}")
            off_molecule = Molecule.from_file(mol_path)
            print(f"Ligand SMILES: {off_molecule.to_smiles()}")

            gaff_generator = GAFFTemplateGenerator(
                molecules=off_molecule, forcefield="gaff-2.11"
            )
            forcefield_files = [
                "amber/protein.ff14SB.xml",
                "amber/tip3p_standard.xml",
                "amber/tip3p_HFE_multivalent.xml",
            ]
            print("Using GAFF-2.11 + AMBER ff14SB for protein-ligand system.")
        else:
            gaff_generator = None
            forcefield_files = ["amber14-all.xml", "amber14/tip3pfb.xml"]
            print("Apo protein mode. Using AMBER14 force field.")

        ff = ForceField(*forcefield_files)
        if gaff_generator:
            ff.registerTemplateGenerator(gaff_generator.generator)

        print("Adding solvating...")
        #modeller.addHydrogens(ff, pH=7.0)
        modeller.addSolvent(
            ff,
            model="tip3p",
            padding=box_padding,
            ionicStrength=0.1 * molar,
        )
        print(
            f"System size: {modeller.topology.getNumAtoms()} atoms, "
            f"{modeller.topology.getNumResidues()} residues"
        )

        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0 * nanometers,
            constraints=HBonds,
            rigidWater=True,
            hydrogenMass=1.5 * amu,
        )
        # NPT barostat
        system.addForce(MonteCarloBarostat(1.0 * atmospheres, 300 * kelvin, 25))

        # Save AMBER topology and coordinates.
        # Use an unconstrained system so ParmEd can extract all bond parameters
        # (constraints=HBonds removes H-bond terms from the System, leaving None types).
        system_for_prmtop = ff.createSystem(
            modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0 * nanometers,
            constraints=None,
            rigidWater=False,
        )
        prmtop_path = os.path.join(self.work_dir, "com.prmtop")
        inpcrd_path = os.path.join(self.work_dir, "com.inpcrd")
        print(f"Saving AMBER files to {self.work_dir} ...")
        structure = pmd.openmm.load_topology(modeller.topology, system_for_prmtop, modeller.positions)
        structure.save(prmtop_path, overwrite=True)
        structure.save(inpcrd_path, overwrite=True)
        print(f"  Saved: com.prmtop, com.inpcrd")

        return modeller.topology, modeller.positions, system

    def run(self, topology=None, positions=None, system=None, job_name="simulation", md_time_ns=2.0):
        # Load from saved AMBER files
        prmtop_path = os.path.join(self.work_dir, "com.prmtop")
        inpcrd_path = os.path.join(self.work_dir, "com.inpcrd")
        print(f"Loading system from: {prmtop_path}")
        prmtop = AmberPrmtopFile(prmtop_path)
        inpcrd = AmberInpcrdFile(inpcrd_path)

        system = prmtop.createSystem(
            nonbondedMethod=PME,
            nonbondedCutoff=1.0 * nanometers,
            constraints=HBonds,
            rigidWater=True,
            hydrogenMass=1.5 * amu,
        )
        system.addForce(MonteCarloBarostat(1.0 * atmospheres, 300 * kelvin, 25))

        topology = prmtop.topology
        positions = inpcrd.positions

        integrator = LangevinMiddleIntegrator(
            300 * kelvin, 1.0 / picosecond, 0.002 * picoseconds
        )
        integrator.setConstraintTolerance(1e-6)

        platform = Platform.getPlatformByName("CUDA")
        properties = {"Precision": "single"}

        simulation = Simulation(topology, system, integrator, platform, properties)
        simulation.context.setPositions(positions)
        if inpcrd.boxVectors is not None:
            simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

        dcd_path = os.path.join(self.results_dir, f"{job_name}.dcd")
        log_path = os.path.join(self.results_dir, f"{job_name}.log")
        chk_path = os.path.join(self.results_dir, f"{job_name}.chk")

        # Report every 20 ps for DCD, every 2 ps for log
        simulation.reporters.append(DCDReporter(dcd_path, 10000))
        simulation.reporters.append(
            StateDataReporter(
                log_path, 1000,
                step=True, time=True, speed=True, progress=True,
                potentialEnergy=True, kineticEnergy=True, temperature=True,
                totalSteps=int(md_time_ns * 500000),
                separator="\t",
            )
        )
        simulation.reporters.append(
            StateDataReporter(
                sys.stdout, 5000,
                step=True, time=True, speed=True, progress=True,
                potentialEnergy=True, temperature=True,
                totalSteps=int(md_time_ns * 500000),
                separator="\t",
            )
        )
        simulation.reporters.append(CheckpointReporter(chk_path, 10000))

        print("\n--- Energy Minimization ---")
        simulation.minimizeEnergy()
        state = simulation.context.getState(getEnergy=True)
        print(f"Potential energy after minimization: {state.getPotentialEnergy()}")

        print("\n--- NVT Equilibration (4 ps) ---")
        simulation.context.setVelocitiesToTemperature(300 * kelvin)
        simulation.step(2000)

        print(f"\n--- Production MD ({md_time_ns} ns) ---")
        n_steps = int(md_time_ns * 500000)
        simulation.step(n_steps)

        print(f"\nSimulation complete. Output files in: {self.results_dir}")
        print(f"  Trajectory: {dcd_path}")
        print(f"  Log:        {log_path}")
        print(f"  Checkpoint: {chk_path}")


def main():
    print("=== OpenMM Protein-Ligand MD Simulation ===\n")

    work_dir = input("Enter working directory (absolute path or subdir name): ").strip()
    if not work_dir:
        work_dir = "test"

    pdb_name = input("Enter PDB filename: ").strip()
    md_time = float(input("Enter simulation time (ns): ").strip())
    job_name = input("Enter job name: ").strip() or "simulation"

    sdf_input = input("Do you have a ligand SDF? (y/n): ").strip().lower()
    sdf_name = None
    if sdf_input.startswith("y"):
        sdf_name = input("Enter SDF filename: ").strip()

    sim = MDSimulator(work_dir)
    topology, positions, system = sim.load_system(pdb_name, sdf_name)
    sim.run(topology, positions, system, job_name, md_time)


if __name__ == "__main__":
    main()
