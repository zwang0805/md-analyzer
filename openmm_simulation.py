# -*- coding: utf-8 -*-
import os
import sys
from openmm import *
from openmm.app import *
from openmm.unit import *
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.toolkit.topology import Molecule
from rdkit import Chem

class MDSimulator:
    def __init__(self, work_dir=None):
        self.work_dir = work_dir or os.getcwd()
        self.results_dir = os.path.join(self.work_dir, 'results')
        self._setup_directories()

    def _setup_directories(self):
        """创建必要的目录"""
        os.makedirs(self.results_dir, exist_ok=True)

    def load_system(self, pdb_name, sdf_name=None, box_padding=1.0*nanometers):
        """
        加载或构建系统。
        支持纯蛋白（Apo）或蛋白-配体复合物。
        """
        pdb_path = os.path.join(self.work_dir, pdb_name)
        
        # 1. 加载拓扑和坐标
        pdb = PDBFile(pdb_path)
        modeller = Modeller(pdb.topology, pdb.positions)
        
        # 2. 处理小分子 (如果提供)
        if sdf_name:
            print("Detected SDF file. Preparing ligand parameters with GAFF...")
            mol_path = os.path.join(self.work_dir, sdf_name)
            off_molecule = Molecule.from_file(mol_path)
            gaff_generator = GAFFTemplateGenerator(molecules=off_molecule)
            forcefield_files = [
                'amber/protein.ff14SB.xml', 
                'amber/tip3p_standard.xml', 
                'amber/tip3p_HFE_multivalent.xml'
            ]
        else:
            print("Apo protein mode. Using standard AMBER force field.")
            gaff_generator = None
            forcefield_files = ['amber14-all.xml', 'amber14/tip3pfb.xml']

        # 3. 构建力场
        ff = ForceField(*forcefield_files)
        if gaff_generator:
            ff.registerTemplateGenerator(gaff_generator.generator)

        # 4. 添加溶剂和氢 (pH 7.0)
        print("Adding hydrogens and solvent...")
        modeller.addHydrogens(ff, pH=7.0)
        modeller.addSolvent(
            ff, 
            model='tip3p', 
            padding=box_padding, 
            ionicStrength=0.1*molar
        )

        # 5. 创建系统
        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0*nanometers,
            constraints=HBonds,
            rigidWater=True,
            hydrogenMass=1.5*amu
        )
        system.addForce(MonteCarloBarostat(1.0*atmospheres, 300*kelvin, 25))
        
        return modeller.topology, modeller.positions, system

    def setup_simulation(self, topology, positions, system, job_name="simulation", md_time_ns=50):
        """
        配置模拟器并运行。
        """
        # 积分器
        integrator = LangevinMiddleIntegrator(
            300*kelvin, 1.0/picosecond, 0.002*picoseconds
        )
        integrator.setConstraintTolerance(1e-6)

        # 平台设置 (CUDA)
        platform = Platform.getPlatformByName('CUDA')
        properties = {'Precision': 'single'}

        # 模拟对象
        simulation = Simulation(
            topology, system, integrator, platform, properties
        )
        simulation.context.setPositions(positions)

        # 报告器
        dcd_path = os.path.join(self.results_dir, f"{job_name}.dcd")
        log_path = os.path.join(self.results_dir, f"{job_name}.log")
        
        simulation.reporters.append(DCDReporter(dcd_path, 10000))
        simulation.reporters.append(StateDataReporter(
            log_path, 1000, step=True, speed=True, progress=True,
            potentialEnergy=True, temperature=True, separator='\t'
        ))
        simulation.reporters.append(CheckpointReporter(
            os.path.join(self.results_dir, f"{job_name}.chk"), 10000
        ))

        # 执行流程
        print("Performing energy minimization...")
        simulation.minimizeEnergy()

        print("Equilibrating...")
        simulation.context.setVelocitiesToTemperature(300*kelvin)
        simulation.step(2000) # 4 ps

        print("Starting production run...")
        n_steps = int(md_time_ns * 500000) # ns / (2fs)
        simulation.step(n_steps)
        print("Simulation completed.")

def main():
    # 交互式输入 (建议在调试时使用，生产环境可改为参数解析)
    work_dir = input("Enter working directory (Enter for current): ").strip() or None
    pdb_name = input("Enter PDB filename: ")
    md_time = float(input("Enter simulation time (ns): "))
    job_name = input("Enter job name: ")
    
    sdf_input = input("Do you have a ligand (SDF)? (y/n): ").lower()
    sdf_name = None
    if sdf_input.startswith('y'):
        sdf_name = input("Enter SDF filename: ")

    # 执行
    sim = MDSimulator(work_dir)
    topology, positions, system = sim.load_system(pdb_name, sdf_name)
    sim.setup_simulation(topology, positions, system, job_name, md_time)

if __name__ == "__main__":
    main()