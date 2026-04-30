# OpenMM Protein-Ligand MD Simulation & Analysis Pipeline

A complete workflow for running all-atom molecular dynamics simulations and post-simulation analysis of apo proteins or protein-ligand complexes using OpenMM and MDAnalysis.

---

## File Overview

| File | Conda env | Description |
|------|-----------|-------------|
| `run_md.py` | `openmm` | Command-line entry point for running MD simulations |
| `openmm_simulation.py` | `openmm` | Core `MDSimulator` class, imported by `run_md.py` |
| `md_analyzer.py` | `mda` | Interactive trajectory analysis tool (`MDAnalyzer` class) |

---

## Part 1 — MD Simulation

### Installation

A conda environment is recommended.

#### 1. Create and activate the environment

```bash
conda create -n openmm python=3.12
conda activate openmm
```

#### 2. Install OpenMM

```bash
conda install -c conda-forge openmm
```

#### 3. Install small-molecule force field dependencies

```bash
# Force field extensions for OpenMM (GAFF, OpenFF, etc.)
conda install -c conda-forge openmmforcefields

# OpenFF Toolkit (small molecule topology handling)
conda install -c conda-forge openff-toolkit

# ParmEd (AMBER format conversion)
conda install -c conda-forge parmed

# RDKit (SDF file parsing)
conda install -c conda-forge rdkit
```

#### 4. Verify installation

```bash
python -c "
import openmm, openmmforcefields, openff.toolkit, parmed, rdkit
print('openmm            :', openmm.__version__)
print('openmmforcefields :', openmmforcefields.__version__)
print('openff-toolkit    :', openff.toolkit.__version__)
print('parmed            :', parmed.__version__)
print('rdkit             :', rdkit.__version__)
"
```

Expected output (versions used in development):

```
openmm            : 8.5.1
openmmforcefields : 0.16.0
openff-toolkit    : 0.18.0
parmed            : 4.3.1
rdkit             : 2025.03.6
```

---

### Input Files

#### Directory layout

The working directory can be an absolute path or a subdirectory name inside `working_directory/`:

```
working_directory/
└── <your_job>/
    ├── pro-lig.pdb   # protein or protein-ligand complex
    └── lig.sdf       # ligand structure (required only for protein-ligand runs)
```

#### PDB file (required)

- Coordinates of the protein or protein-ligand complex.
- Hydrogen atoms may be absent — the program adds them automatically at pH 7.0.
- The ligand residue name inside the PDB can be anything; it is matched to the SDF by chemical structure.
- Recommended sources: MOE, Maestro, or any standard structure preparation pipeline.

#### SDF file (required when a ligand is present)

- 2D or 3D structure of the small molecule.
- The OpenFF Toolkit reads the file; GAFF-2.11 via antechamber assigns atom types and AM1-BCC charges automatically.
- The molecule in the SDF must correspond to the ligand in the PDB.
- Recommended sources: ChemDraw, MarvinSketch, or the PDB ligand library.

---

### Usage

#### Command syntax

```bash
python run_md.py -work_dir <dir> -com <pdb> [-lig <sdf>] [-t <ns>] [-job <name>]
```

#### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `-work_dir` | Yes | — | Working directory (absolute path or subdirectory under `working_directory/`) |
| `-com` | Yes | — | PDB filename (relative to the working directory) |
| `-lig` | No | None | SDF filename; omit for apo-protein simulations |
| `-t` | No | `200.0` | Production simulation length in nanoseconds |
| `-job` | No | PDB basename (no extension) | Prefix for all output files |

#### Examples

**Protein-ligand complex, 200 ns:**

```bash
python run_md.py -work_dir examples -com pro-lig.pdb -lig lig.sdf -t 200
```

**Background run with nohup (recommended for long simulations):**

```bash
nohup /path/to/envs/openmm/bin/python run_md.py \
    -work_dir examples \
    -com pro-lig.pdb \
    -lig lig.sdf \
    -t 200 \
    -job my_simulation \
    > run.log 2>&1 &
```

**Apo protein, 50 ns, with absolute path:**

```bash
python run_md.py -work_dir /absolute/path/to/jobdir -com protein.pdb -t 50
```

---

### Simulation Workflow

```
Input PDB / SDF
      │
      ▼
Load protein coordinates (PDBFile)
      │
      ▼
Parameterize ligand with GAFF-2.11 (if SDF provided)
Force fields: AMBER ff14SB + GAFF-2.11 + TIP3P
      │
      ▼
Add solvent: TIP3P water box (padding = 1.0 nm, 150 mM NaCl)
      │
      ▼
Build OpenMM System (PME, HBonds constraints, HMR 1.5 amu)
      │
      ▼
Save com.prmtop + com.inpcrd  (AMBER format)
      │
      ▼
Reload system from com.prmtop
      │
      ▼
Energy minimization
      │
      ▼
NVT equilibration (4 ps, 300 K)
      │
      ▼
NPT production MD (user-specified length, 300 K, 1 atm)
      │
      ▼
Output DCD + LOG + CHK
```

#### Simulation parameters

| Parameter | Value |
|-----------|-------|
| Protein force field | AMBER ff14SB |
| Ligand force field | GAFF-2.11 |
| Water model | TIP3P |
| Box padding | 1.0 nm |
| Ionic strength | 150 mM NaCl |
| Time step | 2 fs |
| Hydrogen mass repartitioning (HMR) | 1.5 amu |
| Temperature | 300 K (Langevin Middle integrator) |
| Pressure | 1 atm (Monte Carlo barostat) |
| PME cutoff | 1.0 nm |
| Constraint tolerance | 1 × 10⁻⁶ |
| Platform | CUDA (single precision) |

---

### Simulation Output Files

All simulation output is written to `<work_dir>/results/`:

| File | Format | Description |
|------|--------|-------------|
| `<job>.dcd` | DCD | Trajectory, 1 frame saved every 20 ps |
| `<job>.log` | TSV | Thermodynamic log (step, time, potential energy, kinetic energy, temperature, speed), written every 2 ps |
| `<job>.chk` | Binary | Checkpoint file saved every 20 ps; used to restart a simulation |

The following files are also written to the working directory:

| File | Format | Description |
|------|--------|-------------|
| `com.prmtop` | AMBER prmtop | Full force field topology (compatible with cpptraj / AmberTools / md_analyzer.py) |
| `com.inpcrd` | AMBER inpcrd | Initial coordinates and box vectors |

---

### Troubleshooting (Simulation)

**CUDA not available**

Change the platform in the `run()` method of `openmm_simulation.py`:

```python
platform = Platform.getPlatformByName("CPU")   # or "OpenCL"
properties = {}
```

**Ligand parameterization fails**

Verify that the SDF file has a valid, complete bond structure:

```python
from rdkit import Chem
mol = Chem.MolFromMolFile("lig.sdf")
print(Chem.MolToSmiles(mol))
```

**Restarting from a checkpoint**

Load the checkpoint in `openmm_simulation.py` inside `run()`, before the minimization step:

```python
simulation.loadCheckpoint(chk_path)
```

---

---

## Part 2 — Trajectory Analysis

`md_analyzer.py` provides an interactive menu for analysing simulation output from Part 1.  
It uses **MDAnalysis**, **ProLIF**, and **scikit-learn** and requires the separate `mda` conda environment.

---

### Installation (mda environment)

#### 1. Create and activate the environment

```bash
conda create -n mda python=3.12
conda activate mda
```

#### 2. Install dependencies

```bash
conda install -c conda-forge mdanalysis
conda install -c conda-forge prolif
conda install -c conda-forge pandas numpy matplotlib scikit-learn
```

#### 3. Verify installation

```bash
python -c "
import MDAnalysis, prolif, pandas, sklearn
print('MDAnalysis:', MDAnalysis.__version__)
print('prolif    :', prolif.__version__)
print('pandas    :', pandas.__version__)
print('sklearn   :', sklearn.__version__)
"
```

Expected output (versions used in development):

```
MDAnalysis: 2.10.0
prolif    : 2.1.0
pandas    : 3.0.2
sklearn   : 1.8.0
```

---

### Input Files

`md_analyzer.py` reads the files produced by Part 1:

| File | Location | Description |
|------|----------|-------------|
| `com.prmtop` | `<work_dir>/` | AMBER topology with all force field parameters |
| `<job>.dcd` | `<work_dir>/results/` | DCD trajectory from `run_md.py` |

Any topology format supported by MDAnalysis (`.pdb`, `.psf`, `.gro`, `.prmtop` …) and any trajectory format (`.dcd`, `.xtc`, `.nc` …) are accepted.

---

### Usage

#### Interactive mode (recommended)

```bash
/path/to/envs/mda/bin/python md_analyzer.py
```

The script prompts for the working directory name, auto-detects the topology and trajectory files, then presents an analysis menu.

#### Programmatic mode

```python
from md_analyzer import MDAnalyzer

ana = MDAnalyzer(directory='examples')
ana.read_pdb_dcd('com.prmtop', 'pro-lig.dcd', apo=False,
                 remove_subunits='no', dt='20.0')
ana.rmsd(rmsd_step=10)
ana.rmsf(selection='resname UNK', ligand=True)
```

Key parameters for `read_pdb_dcd()`:

| Parameter | Description |
|-----------|-------------|
| `pdb` | Topology filename (relative to `<work_dir>/`) |
| `dcd` | Trajectory filename (relative to `<work_dir>/results/`) |
| `apo` | `True` for apo protein (skips ligand PBC correction) |
| `remove_subunits` | Selection string to discard atoms (e.g. `'resname HOH'`); pass `'no'` to skip |
| `solv_selection` | Comma-separated solvent resnames (e.g. `'HOH,WAT'`); pass `'no'` to skip |
| `lig_selection` | Comma-separated ligand resnames (e.g. `'UNK'`); pass `'no'` to skip |
| `dt` | Override trajectory timestep (ps); leave empty to use value from file |

---

### Analysis Functions

#### 1 — RMSD

Calculates backbone, Cα, whole-protein, and (optionally) ligand RMSD vs. the first frame.

```python
ana.rmsd(rmsd_step=10)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rmsd_step` | `10` | Analyse every Nth frame |

Output: `<work_dir>/analysis/<pdb>_rmsd.png` and `_rmsd.csv`

---

#### 2 — Extract Frames

Saves evenly-spaced frames within a time window as a multi-model PDB file.

```python
ana.extract_complex(start=0, end=500, slices=50)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start` | `0` | Start time (same units as `dt`) |
| `end` | `0` | End time; `-1` for trajectory end |
| `slices` | `50` | Number of frames to save |

Output: `<work_dir>/analysis/<pdb>_frames_<N>.pdb`

---

#### 3 — Distance Analysis

Calculates the distance between two atom groups over the trajectory and plots a moving average.

```python
ana.distance_analysis(
    atom_label='protein and resid 30 and name CZ, ligand and name S1',
    dist_step=10,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `atom_label` | (interactive) | Two MDAnalysis selection strings separated by `,`. Use `ligand` as a shorthand for the auto-detected ligand. |
| `dist_step` | `10` | Analyse every Nth frame |

Output: `<work_dir>/analysis/<pdb>_dist_<suffix>.png` and `_dist_<suffix>.csv`

---

#### 4 — RMSF

Calculates per-atom RMSF for the ligand, or per-residue RMSF for the protein.

```python
# Ligand
ana.rmsf(selection='resname UNK', ligand=True, rmsf_step=10)

# Protein Cα
ana.rmsf(selection='protein and name CA', ligand=False,
         align_str='protein and name CA', rmsf_step=10)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `selection` | (interactive) | MDAnalysis selection string |
| `ligand` | `True` | `True` → per-atom plot; `False` → per-residue plot |
| `align_str` | `'protein and name CA'` | Selection used for trajectory alignment before RMSF |
| `rmsf_step` | `10` | Analyse every Nth frame |

Output: `<work_dir>/analysis/<pdb>_ligand_rmsf.png/.csv` or `_rmsf.png/.csv`

---

#### 5 — Protein-Ligand Interaction Fingerprint

Uses **ProLIF** to calculate and visualise interaction fingerprints (H-bonds, hydrophobic contacts, π-stacking, etc.) across the trajectory.

```python
ana.pro_lig_int(selection_str='resname UNK', lig=True)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `selection_str` | (auto: `ligand_str`) | Ligand or peptide selection string |
| `lig` | `True` | `True` → small-molecule ligand; `False` → protein segment |

Output: `<work_dir>/analysis/<pdb>_fingerprint_<suffix>.pkl/.csv` and `_prolig_<suffix>.png`

> **Note:** ProLIF requires all hydrogens to be present and bond orders to be determinable.
> For PDB-based topologies, elements are guessed automatically.
> For prmtop files, elements are read directly from the file.

---

#### 6 — Dihedral Angle

Calculates the dihedral angle of four atoms and plots its distribution.

```python
# Ligand (specify atom IDs)
ana.dihedral_angle(atom_label='3045,3046,3049,3050', dist_step=5, ligand=True)

# Protein residue (MDAnalysis selection string)
ana.dihedral_angle(
    atom_label='resid 10 and (name N or name CA or name CB or name C)',
    dist_step=5,
    ligand=False,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `atom_label` | (interactive) | Four atom IDs (ligand) or one MDAnalysis selection yielding 4 atoms (protein) |
| `dist_step` | `10` | Analyse every Nth frame |
| `ligand` | `True` | `True` → interpret label as atom IDs; `False` → interpret as selection string |

Output: `<work_dir>/analysis/<pdb>_dihedral_<suffix>.png` and `_dihedral_<suffix>.csv`

---

#### 7 — Pairwise RMSD Heatmap

Computes and plots a symmetric N×N RMSD matrix between all trajectory frames for the protein and ligand separately.

```python
ana.heatmap()
```

Output: `<work_dir>/analysis/<pdb>_heatmap.png`

---

#### 8 — Secondary Structure (DSSP)

Assigns DSSP secondary structure (`-` loop, `H` helix, `E` sheet) to every protein residue at each frame.
Capping residues (ACE, NME, etc.) are excluded automatically.

```python
ana.DSSP(dssp_step=10)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `residues` | `'all'` | Residue index range in `start-end` format (e.g. `'10-50'`), or `'all'` |
| `dssp_step` | `50` | Analyse every Nth frame |

Output: `<work_dir>/analysis/<pdb>_dssp.csv`

---

#### 9 — Clustering

Clusters trajectory frames using **MDAnalysis encore** (K-means by default) and saves a representative PDB structure for each cluster.

```python
from unittest.mock import patch
with patch('builtins.input', return_value='2'):   # '2' = CA atoms
    ana.cluster_md_trajectory(start=0, end=1000, jobname='run1', n_clusters=5)
```

When called interactively, the script prompts for the atom selection:

```
Atom selection for clustering:
  (1) Pocket residues + ligand
  (2) CA atoms (default)
  (3) Custom selection string
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start` | `0` | Start time |
| `end` | `0` | End time; `-1` for trajectory end |
| `jobname` | `'clustering'` | Output subdirectory name |
| `n_clusters` | `5` | Number of clusters |
| `clustering_method` | `encore.KMeans` | Any MDAnalysis encore clustering method |
| `selection` | `'name CA'` | Fallback selection if interactive prompt is bypassed |

Output: `<work_dir>/analysis/<jobname>/cluster_<N>_rep.pdb` for each cluster

---

### Analysis Output Summary

All output files are written to `<work_dir>/analysis/`:

| Analysis | Files |
|----------|-------|
| RMSD | `*_rmsd.png`, `*_rmsd.csv` |
| Extract frames | `*_frames_<N>.pdb` |
| Distance | `*_dist_<atoms>.png`, `*_dist_<atoms>.csv` |
| RMSF (ligand) | `*_ligand_rmsf.png`, `*_ligand_rmsf.csv` |
| RMSF (protein) | `*_rmsf.png`, `*_rmsf.csv` |
| Fingerprint | `*_fingerprint_<lig>.pkl`, `*_fingerprint_<lig>.csv`, `*_prolig_<lig>.png` |
| Dihedral | `*_dihedral_<atoms>.png`, `*_dihedral_<atoms>.csv` |
| Heatmap | `*_heatmap.png` |
| DSSP | `*_dssp.csv` |
| Clustering | `<jobname>/cluster_<N>_rep.pdb` |

---

### Troubleshooting (Analysis)

**DSSP fails with "unequal numbers of N/CA/C/O atoms"**

This happens when non-standard residues (caps, ligands) are included.
`md_analyzer.py` excludes `ACE`, `NME`, and common caps automatically.
If the error persists, check for other non-standard residues with `ana.universe_info()`.

**ProLIF fails to convert molecule**

ProLIF requires bond order information.  For PDB topologies, run:

```python
elements = MDAnalysis.topology.guessers.guess_types(ana.u.atoms.names)
ana.u.add_TopologyAttr('elements', elements)
```

For prmtop topologies (e.g. `com.prmtop`), elements are read automatically and no extra steps are needed.

**Trajectory timestep appears wrong**

Call `read_pdb_dcd()` with `dt='<value_in_ps>'` to override the value from the file header, or enter the correct value when prompted interactively.
