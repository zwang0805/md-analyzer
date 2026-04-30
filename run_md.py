# -*- coding: utf-8 -*-
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openmm_simulation import MDSimulator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OpenMM protein-ligand MD simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-work_dir", required=True,
        help="Working directory (absolute path or subdir name under working_directory/)",
    )
    parser.add_argument(
        "-com", required=True, metavar="PDB",
        help="PDB filename of the protein or protein-ligand complex",
    )
    parser.add_argument(
        "-lig", default=None, metavar="SDF",
        help="SDF filename of the ligand (omit for apo protein)",
    )
    parser.add_argument(
        "-t", type=float, default=200.0, metavar="NS",
        help="Simulation time in nanoseconds",
    )
    parser.add_argument(
        "-job", default=None, metavar="NAME",
        help="Job name prefix for output files (default: PDB basename without extension)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    job_name = args.job or os.path.splitext(args.com)[0]

    print(f"Work directory : {args.work_dir}")
    print(f"PDB            : {args.com}")
    print(f"Ligand SDF     : {args.lig or 'None (apo mode)'}")
    print(f"Simulation time: {args.t} ns")
    print(f"Job name       : {job_name}")
    print()

    sim = MDSimulator(args.work_dir)
    topology, positions, system = sim.load_system(args.com, args.lig)
    sim.run(topology, positions, system, job_name=job_name, md_time_ns=args.t)


if __name__ == "__main__":
    main()
pyt