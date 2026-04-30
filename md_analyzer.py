# -*- coding: utf-8 -*-
import os
import MDAnalysis as mda
import pandas as pd
import numpy as np
from MDAnalysis.analysis import rms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from MDAnalysis.analysis import align, diffusionmap
from MDAnalysis import transformations
from MDAnalysis.analysis import distances
import prolif as plf
import MDAnalysis.analysis.encore as encore
from MDAnalysis.transformations.base import TransformationBase


class GroupHug(TransformationBase):
    """On-the-fly transformation that keeps all protein chains inside the same
    periodic image as the reference chain, fixing broken-chain artefacts from PBC."""

    def __init__(self, center, *others):
        super().__init__(max_threads=1, parallelizable=True)
        self.c = center
        self.o = others

    @staticmethod
    def calc_restoring_vec(ag1, ag2):
        box = ag1.dimensions[:3]
        dist = ag1.center_of_mass() - ag2.center_of_mass()
        return box * np.rint(dist / box)

    def _transform(self, ts):
        for i in self.o:
            rvec = self.calc_restoring_vec(self.c, i)
            i.translate(+rvec)
        return ts


class MDAnalyzer:
    def __init__(self, directory=''):
        self.path = os.path.split(os.path.realpath(__file__))[0]
        if directory == '':
            self.working_directory = os.path.join(self.path, 'working_directory')
        else:
            self.working_directory = os.path.join(self.path, 'working_directory', directory)
        os.makedirs(os.path.join(self.working_directory, 'analysis'), exist_ok=True)

        self.solvent_str = '(resname WAT or resname SOL or resname HOH)'
        self.ion_str = (
            '(resname NA or resname CL or resname NA? or resname CL? '
            'or resname MG or resname MG* or resname MN*)'
        )
        self.ligand_str = '(not (%s or %s or protein))' % (self.solvent_str, self.ion_str)
        self.delete_units = ''

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def universe_info(self):
        """Print a summary of all residue types in the universe."""
        # Iterate residues (not atoms) for efficiency
        all_resnames = {r.resname for r in self.u.residues}
        pro_resnames = {r.resname for r in self.u.select_atoms('protein').residues}
        other_resnames = all_resnames - pro_resnames
        print('Segment IDs :', set(self.u.segments.segids))
        print('All residues:', all_resnames)
        print('Protein     :', pro_resnames)
        print('Non-protein :', other_resnames)

    def get_frames_by_time(self, start_time, end_time):
        """Return (start_frame, end_frame_exclusive) corresponding to a time range.

        Parameters
        ----------
        start_time, end_time : float
            Times in the same units as self.dt.  Pass -1 for end_time to select
            until the last frame.

        Returns
        -------
        tuple(int, int)
            ``(start_idx, end_idx_exclusive)`` suitable for use in a Python slice
            ``trajectory[start:end]`` or ``range(start, end)``.
        """
        dt = self.dt
        t0 = self.u.trajectory[0].time
        total_frames = len(self.u.trajectory)

        start_idx = max(0, int(round((float(start_time) - t0) / dt)))

        if end_time == -1:
            end_idx_excl = total_frames
        else:
            end_idx_incl = min(total_frames - 1,
                               int(round((float(end_time) - t0) / dt)))
            end_idx_excl = end_idx_incl + 1

        if start_idx >= end_idx_excl:
            print(f"Warning: start frame ({start_idx}) >= end frame ({end_idx_excl}). "
                  "Check the requested time range.")
            return (0, 0)

        total_time = dt * total_frames
        print(f'dt={dt}, total frames={total_frames}, total time={total_time}')
        print(f'start_time={start_time} → frame {start_idx}')
        print(f'end_time={end_time}   → frame {end_idx_excl - 1} (exclusive end={end_idx_excl})')
        return (start_idx, end_idx_excl)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def read_pdb_dcd(self, pdb, dcd, apo=False, remove_subunits='',
                     solv_selection='', lig_selection='', dt=''):
        """Load topology and trajectory, apply PBC corrections on-the-fly.

        Parameters
        ----------
        pdb : str
            Topology filename (any format MDAnalysis accepts: .pdb, .prmtop, .gro, .psf …).
            Resolved relative to self.working_directory.
        dcd : str
            Trajectory filename (any format MDAnalysis accepts: .dcd, .xtc …).
            Resolved relative to ``self.working_directory/results/``.
        apo : bool
            True for apo-protein (no ligand PBC correction needed).
        remove_subunits : str
            Selection string for atoms to discard before analysis (e.g. 'resname HOH').
            Pass 'no' to skip the interactive prompt entirely.
        solv_selection, lig_selection : str
            Override for solvent / ligand selection strings.  Pass 'no' to skip.
        dt : str or float
            Override the trajectory timestep. If empty, the value from the
            trajectory header is used. Units must match the time arguments
            passed to get_frames_by_time() / cluster_md_trajectory() etc.
        """
        from MDAnalysis.topology.guessers import guess_types

        topo_path = os.path.join(self.working_directory, pdb)
        traj_path = os.path.join(self.working_directory, 'results', dcd)
        self.u = mda.Universe(topo_path, traj_path)

        # Warn about suspicious water residue 1
        wat = self.u.select_atoms('resname HOH and resid 1')
        if len(wat) > 3:
            print(f'Warning: residue HOH/1 has {len(wat)} atoms — consider removing it.')

        self.dt = self.u.trajectory.dt
        if dt == '':
            dt_query = input(
                f'Trajectory timestep is {self.dt} ps.  '
                'Press Enter to keep, or type a new value: '
            ).strip()
            if dt_query:
                self.dt = float(dt_query)
        else:
            self.dt = float(dt)

        self.pdb = pdb
        self.apo = apo
        self.dcd = dcd

        if remove_subunits != 'no':
            self._remove_subunits(delete_units=remove_subunits,
                                  solvent_selection=solv_selection,
                                  ligand_selection=lig_selection)

        if '.gro' in pdb:
            self.u.atoms.guess_bonds()
            elements = guess_types(self.u.atoms.types)
            self.u.add_TopologyAttr('elements', elements)

        # Build reference for alignment (first-frame protein)
        ref_universe = mda.Universe(topo_path)
        if self.delete_units:
            reference = ref_universe.select_atoms(
                'protein and (not (%s))' % self.delete_units)
        else:
            reference = ref_universe.select_atoms('protein')

        # Gather protein chain segments for GroupHug
        prot_chain_list = [
            chain.atoms for chain in self.u.select_atoms('protein').segments
            if chain.atoms.n_atoms > 0
        ]
        print(f'Protein chains: {[c.segids for c in self.u.select_atoms("protein").segments]}')
        prot_group = GroupHug(prot_chain_list[0], *prot_chain_list[1:])

        # On-the-fly transformation pipeline
        self.workflow = (
            mda.transformations.unwrap(self.u.atoms, max_threads=10),
            prot_group,
            mda.transformations.center_in_box(
                self.u.select_atoms('protein'), center='geometry', max_threads=10),
            mda.transformations.wrap(
                self.u.select_atoms('protein or (%s)' % self.ligand_str),
                compound='fragments', max_threads=10),
            mda.transformations.wrap(
                self.u.select_atoms(self.solvent_str),
                compound='residues', max_threads=10),
            mda.transformations.fit_rot_trans(
                self.u.select_atoms('protein'), reference, max_threads=10),
        )
        self.u.trajectory.add_transformations(*self.workflow)

    def _remove_subunits(self, delete_units='', solvent_selection='', ligand_selection=''):
        """Remove unwanted atom groups and update the Universe and selection strings."""
        self.universe_info()

        if delete_units == '':
            delete_units = input(
                'Remove atoms from the Universe?\n'
                '  (1) No\n'
                '  or type a selection string (e.g. "chainID B")\n'
                'See https://userguide.mdanalysis.org/stable/selections.html\n'
            )
        if delete_units != '1':
            self.delete_units = delete_units
            selection = self.u.select_atoms('not (%s)' % delete_units)

            tag = delete_units.replace(' ', '_')
            new_pdb = self.pdb.split('.')[0] + f'_no_{tag}.pdb'
            ext = self.dcd.split('.')[-1]
            new_dcd = self.dcd.split('.')[0] + f'_no_{tag}.{ext}'
            self.pdb = new_pdb
            self.dcd = new_dcd

            print(f'Writing filtered topology → {self.pdb}')
            print(f'Writing filtered trajectory → {self.dcd}')

            if 'psf' in self.pdb:
                from MDAnalysis.coordinates.PSF import PSFWriter
                with PSFWriter(os.path.join(self.working_directory, self.pdb)) as psf:
                    psf.write(selection)
            else:
                selection.write(os.path.join(self.working_directory, self.pdb))

            out_traj = os.path.join(self.working_directory, 'results', self.dcd)
            with mda.Writer(out_traj, selection.n_atoms) as W:
                for ts in self.u.trajectory:
                    W.write(selection)

            self.u = mda.Universe(
                os.path.join(self.working_directory, self.pdb),
                os.path.join(self.working_directory, 'results', self.dcd),
            )
            print(f'Reloaded Universe: {self.pdb}, {self.dcd}')
            self.universe_info()

        # Update solvent string
        if solvent_selection != 'no':
            if solvent_selection == '':
                solvent_selection = input(
                    f'Current solvent definition: {self.solvent_str}\n'
                    'Keep (1) or provide comma-separated resnames (e.g. TIP3,SOL): '
                )
            if solvent_selection != '1':
                resnames = [s.strip() for s in solvent_selection.split(',')]
                self.solvent_str = '(' + ' or '.join(f'resname {r}' for r in resnames) + ')'
                self.ligand_str = '(not (%s or %s or protein))' % (
                    self.solvent_str, self.ion_str)
                print(f'Solvent definition updated: {self.solvent_str}')

        # Update ligand string
        if not self.apo and ligand_selection != 'no':
            if ligand_selection == '':
                ligand_selection = input(
                    f'Current ligand definition: {self.ligand_str}\n'
                    'Keep (1) or provide comma-separated resnames (e.g. UNL,LIG): '
                )
            if ligand_selection != '1':
                resnames = [s.strip() for s in ligand_selection.split(',')]
                self.ligand_str = '(' + ' or '.join(f'resname {r}' for r in resnames) + ')'
                print(f'Ligand definition updated: {self.ligand_str}')

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def rmsd(self, rmsd_step=10):
        """Calculate backbone / CA / protein (and ligand) RMSD vs. first frame."""
        if self.apo:
            R = rms.RMSD(self.u, select='backbone',
                         groupselections=['name CA', 'protein'])
            R.run(step=rmsd_step)
            cols = ['backbone', 'C-alphas', 'protein']
        else:
            R = rms.RMSD(self.u, select='backbone',
                         groupselections=['name CA', 'protein', self.ligand_str])
            R.run(step=rmsd_step)
            cols = ['backbone', 'C-alphas', 'protein', 'ligand']

        rmsd_df = pd.DataFrame(R.rmsd[:, 2:], columns=cols,
                               index=R.rmsd[:, 0] * self.dt)
        rmsd_df.index.name = 'Time'
        print(rmsd_df)

        ax = rmsd_df.plot(title='RMSD')
        fig = ax.get_figure()
        out_png = os.path.join(self.working_directory, 'analysis',
                               f'{self.pdb}_rmsd.png')
        out_csv = os.path.join(self.working_directory, 'analysis',
                               f'{self.pdb}_rmsd.csv')
        fig.savefig(out_png)
        rmsd_df.to_csv(out_csv)
        plt.close('all')
        print(f'RMSD saved to {out_png}')

    def rmsf(self, selection='', ligand=True, align_str='protein and name CA', rmsf_step=10):
        """Calculate per-atom RMSF for the ligand or per-residue RMSF for the protein."""
        if selection == '':
            choice = input('(1) Ligand atoms  (2) Protein/other selection\n')
            if choice == '1':
                selection = self.ligand_str
                ligand = True
            else:
                selection = input('Selection string (e.g. "protein and name CA"): ')
                ligand = False

        pdbname = self.pdb.split('.')[0]
        protein_ag = self.u.select_atoms(selection)

        prealigner = align.AlignTraj(self.u, self.u, select=align_str,
                                     in_memory=True).run(step=rmsf_step)
        ref_coords = self.u.trajectory.timeseries(asel=protein_ag).mean(axis=1)
        ref = mda.Merge(protein_ag).load_new(ref_coords[:, None, :], order='afc')
        align.AlignTraj(self.u, ref, select=selection, in_memory=True).run(step=rmsf_step)

        ag = self.u.select_atoms(selection)
        R = rms.RMSF(ag).run(step=rmsf_step)

        if ligand:
            n = len(ag)
            df = pd.DataFrame({'atom_index': range(1, n + 1),
                                'rmsf': R.results.rmsf[:n]})
            out_csv = os.path.join(self.working_directory, 'analysis',
                                   f'{pdbname}_ligand_rmsf.csv')
            out_png = os.path.join(self.working_directory, 'analysis',
                                   f'{pdbname}_ligand_rmsf.png')
            df.to_csv(out_csv, index=False)
            plt.plot(df['atom_index'], df['rmsf'], 'ok-')
            plt.xlabel('Atom index')
            plt.ylabel(r'RMSF ($\AA$)')
        else:
            df = pd.DataFrame({'residue_id': ag.resids, 'atom': ag.names,
                                'rmsf': R.results.rmsf})
            out_csv = os.path.join(self.working_directory, 'analysis',
                                   f'{pdbname}_rmsf.csv')
            out_png = os.path.join(self.working_directory, 'analysis',
                                   f'{pdbname}_rmsf.png')
            df.to_csv(out_csv, index=False)
            plt.plot(ag.resids, R.results.rmsf, label='RMSF')
            plt.xlabel('Residue number')
            plt.ylabel(r'RMSF ($\AA$)')
            plt.legend()

        df.to_csv(out_csv, index=False)
        plt.savefig(out_png, dpi=300)
        plt.close('all')
        print(f'RMSF saved to {out_png}')

    def distance_analysis(self, atom_label='', dist_step=10):
        """Calculate the distance between two atom groups over the trajectory.

        Parameters
        ----------
        atom_label : str
            Two comma-separated MDAnalysis selection strings.
            Use the keyword ``ligand`` as a shorthand for the auto-detected ligand.
            Example: ``resid 30 and name CZ, ligand and name S25``
        dist_step : int
            Calculate distance every ``dist_step`` frames.
        """
        if atom_label == '':
            atom_label = input(
                'Enter two selection strings separated by "," '
                '(use "ligand" for the small molecule):\n'
                'Ref: https://userguide.mdanalysis.org/stable/selections.html\n'
            )

        atom_label2 = atom_label.replace('ligand', self.ligand_str)
        atom1_exp, atom2_exp = atom_label2.split(',', 1)
        atom1 = self.u.select_atoms(atom1_exp.strip())
        atom2 = self.u.select_atoms(atom2_exp.strip())

        dist = []
        time_ids = []
        for idx, ts in enumerate(self.u.trajectory[::dist_step]):
            d = distances.distance_array(
                atom1.positions, atom2.positions, box=self.u.dimensions)[0, 0]
            dist.append(d)
            time_ids.append(idx * self.dt * dist_step)
            print(f'{time_ids[-1]:.1f}  {d:.3f} Å')

        df = pd.DataFrame({'time': time_ids, 'distance (A)': dist})

        window = 10
        dist_smooth = np.convolve(dist, np.ones(window) / window, mode='valid')
        half = window // 2

        plt.plot(time_ids, dist, alpha=0.5, label='raw')
        plt.plot(time_ids[half: -half + 1 if half > 1 else None],
                 dist_smooth, color='red', linewidth=2, label='moving avg')
        plt.xlabel('Time')
        plt.ylabel(r'Distance ($\AA$)')
        plt.title('Distance (mean=%.2f Å, std=%.2f Å)' % (
            np.mean(dist), np.std(dist)))
        plt.legend()

        suffix = atom_label.replace(' ', '_').replace(',', '_')
        out_png = os.path.join(self.working_directory, 'analysis',
                               f'{self.pdb}_dist_{suffix}.png')
        out_csv = os.path.join(self.working_directory, 'analysis',
                               f'{self.pdb}_dist_{suffix}.csv')
        plt.savefig(out_png, dpi=300)
        plt.clf()
        plt.close('all')
        df.to_csv(out_csv, index=False)
        print(f'Distance analysis saved to {out_png}')
        return df

    def dihedral_angle(self, atom_label='', dist_step=10, ligand=True):
        """Calculate and plot the dihedral angle distribution."""
        if ligand:
            for atom in self.u.select_atoms(self.ligand_str):
                print(f'  id={atom.id}  name={atom.name}  '
                      f'resname={atom.resname}{atom.resid}  pos={atom.position}')
            if atom_label == '':
                atom_label = input('Atom IDs for dihedral (e.g. 2,3,4,5):\n')
            id1, id2, id3, id4 = atom_label.split(',')
            ag_str = 'id %s or id %s or id %s or id %s' % (id1, id2, id3, id4)
        else:
            if atom_label == '':
                atom_label = input(
                    'Selection for dihedral '
                    '(e.g. resid 93 and (name CA or name CB or name CG or name C)):\n'
                )
            ag_str = atom_label

        from MDAnalysis.analysis.dihedrals import Dihedral
        ag = self.u.select_atoms(ag_str)
        R = Dihedral([ag]).run(step=dist_step)
        n_frames = len(self.u.trajectory[::dist_step])
        time_ids = [i * self.dt * dist_step for i in range(n_frames)]

        dihedrals = R.results.angles.ravel()
        df = pd.DataFrame({'time': time_ids, 'dihedral angle': dihedrals})

        suffix = atom_label.replace(' ', '_').replace(',', '_')
        out_csv = os.path.join(self.working_directory, 'analysis',
                               f'{self.pdb}_dihedral_{suffix}.csv')
        out_png = os.path.join(self.working_directory, 'analysis',
                               f'{self.pdb}_dihedral_{suffix}.png')
        df.to_csv(out_csv, index=False)

        plt.hist(dihedrals, bins=60, density=True, alpha=0.6, color='g')
        plt.title('Dihedral angle distribution')
        plt.xlabel('Dihedral angle (°)')
        plt.ylabel('Density')
        plt.savefig(out_png, dpi=300)
        plt.clf()
        plt.close('all')
        print(f'Dihedral analysis saved to {out_png}')

    def calculate_angle_trajectory(self, atom_label=''):
        """Calculate and plot a three-atom angle over the trajectory."""
        from MDAnalysis.lib.distances import calc_angles

        if atom_label == '':
            atom_label = input('Three atom indices (e.g. 20,21,22):\n')
        atom_indices = [int(a) for a in atom_label.split(',')]

        for idx in atom_indices:
            if idx < 0 or idx >= len(self.u.atoms):  # FIX: was len(u.atoms)
                print(f'Error: atom index {idx} out of range (0-{len(self.u.atoms)-1})')
                return None, None

        a1, a2, a3 = atom_indices
        print(f'Calculating angle {a1}-{a2}-{a3}  '
              f'({self.u.atoms[a1].name}-{self.u.atoms[a2].name}-{self.u.atoms[a3].name})')

        times, angles = [], []
        for ts in self.u.trajectory:
            angle_rad = calc_angles(
                self.u.atoms[a1].position,
                self.u.atoms[a2].position,
                self.u.atoms[a3].position,
                box=self.u.dimensions,
            )
            angles.append(np.degrees(angle_rad).flatten()[0])
            times.append(ts.time)

        times = np.array(times)
        angles = np.array(angles)

        print(f'Angle stats — mean: {np.mean(angles):.2f}°  '
              f'std: {np.std(angles):.2f}°  '
              f'min: {np.min(angles):.2f}°  '
              f'max: {np.max(angles):.2f}°')

        df = pd.DataFrame({'time': times, 'angle': angles})
        out_csv = os.path.join(self.working_directory, 'analysis',
                               f'{self.pdb}_angle.csv')
        out_png = os.path.join(self.working_directory, 'analysis',
                               f'{self.pdb}_angle.png')
        df.to_csv(out_csv, index=False)

        stats_text = f'mean={np.mean(angles):.2f}°\nstd={np.std(angles):.2f}°'
        plt.figure(figsize=(10, 6))
        plt.plot(times, angles, 'b-', linewidth=1, alpha=0.7)
        plt.xlabel('Time (ps)')
        plt.ylabel('Angle (°)')
        plt.title(f'Atom {a1}-{a2}-{a3} angle vs time')
        plt.grid(True, alpha=0.3)
        plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8),
                     verticalalignment='top', fontsize=10)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.clf()
        plt.close('all')
        print(f'Angle analysis saved to {out_png}')

    def extract_complex(self, start=0, end=0, slices=50):
        """Save ``slices`` evenly-spaced frames within [start, end] as a multi-model PDB."""
        start_frame, end_frame = self.get_frames_by_time(start_time=start, end_time=end)
        selected_frames = list(range(start_frame, end_frame))

        pro = self.u.select_atoms('all')
        n = len(selected_frames)
        step = max(1, n // slices)
        frames_to_write = selected_frames[::step]

        out = os.path.join(self.working_directory, 'analysis',
                           f'{self.pdb}_frames_{slices}.pdb')
        pro.write(out, frames=frames_to_write)
        print(f'Saved {len(frames_to_write)} frames to {out}')

    def DSSP(self, residues='all', dssp_step=50):
        """Run DSSP secondary structure analysis over the trajectory.

        Only standard amino acid residues (with complete N/CA/C/O backbone) are
        analysed.  Capping residues such as ACE and NME are excluded automatically.
        """
        from MDAnalysis.analysis.dssp import DSSP

        # Select only residues that carry a full backbone; caps (ACE, NME) lack
        # some backbone atoms and cause a ValueError in MDAnalysis DSSP.
        protein_ag = self.u.select_atoms(
            'protein and not resname ACE NME ACE2 FOR NH2'
        )

        start_res, end_res = 0, None
        if '-' in str(residues):
            s, e = residues.split('-')
            start_res, end_res = int(s), int(e)

        run = DSSP(protein_ag).run(step=dssp_step)
        ssa = run.results.dssp[:, start_res:end_res]

        pdbname = self.pdb.split('.')[0]
        df = pd.DataFrame(ssa)
        df.columns = run.results.resids[start_res:end_res]

        out = os.path.join(self.working_directory, 'analysis',
                           f'{pdbname}_dssp.csv')
        df.to_csv(out)
        print("DSSP legend: '-' = loop, 'H' = helix, 'E' = sheet")
        print(f'DSSP saved to {out}')

    def heatmap(self):
        """Plot a pairwise RMSD heatmap for protein and ligand."""
        dist_protein = self._rmsd_dist_matrix(self.u, 'protein')
        dist_ligand = self._rmsd_dist_matrix(self.u, self.ligand_str)
        vmax = max(dist_ligand.max(), dist_protein.max())

        fig, axes = plt.subplots(1, 2)
        fig.suptitle('Pairwise RMSD between frames')
        for ax, mat, label in zip(axes, [dist_protein, dist_ligand],
                                  ['Protein', 'Ligand']):
            im = ax.imshow(mat, cmap='viridis', vmin=0, vmax=vmax)
            ax.set_title(label)
            ax.set_xlabel('Frame')
            ax.set_ylabel('Frame')
        fig.colorbar(im, ax=axes, orientation='horizontal',
                     fraction=0.1, label='RMSD (Å)')
        out = os.path.join(self.working_directory, 'analysis',
                           f'{self.pdb}_heatmap.png')
        fig.savefig(out)
        plt.close()
        print(f'Heatmap saved to {out}')

    def _rmsd_dist_matrix(self, universe, selection):
        """Return the symmetric pairwise-RMSD matrix for a given selection."""
        pw = diffusionmap.DistanceMatrix(universe, select=selection)
        pw.run()
        return pw.results.dist_matrix

    def pro_lig_int(self, selection_str='', lig=True):
        """Calculate protein-ligand interaction fingerprint with ProLIF.

        Requires all heavy atoms and hydrogens in the topology.
        """
        from MDAnalysis.topology.guessers import guess_types

        if selection_str == '':
            ligand_selection = self.u.select_atoms(self.ligand_str)
        else:
            ligand_selection = self.u.select_atoms(selection_str)

        if lig:
            if '.pdb' in self.pdb or '.psf' in self.pdb:
                print('Guessing elements from atom names…')
                elements = guess_types(self.u.atoms.names)
                self.u.add_TopologyAttr('elements', elements)
            ligand_mol = plf.Molecule.from_mda(ligand_selection)
            pocket_selection = self.u.select_atoms(
                'protein and byres around 6.0 group ligand',
                ligand=ligand_selection,
            )
        else:
            if '.pdb' in self.pdb:
                ligand_selection.guess_bonds()
            pocket_selection = self.u.select_atoms(
                'protein and not group peptide', peptide=ligand_selection)
            if '.pdb' in self.pdb:
                pocket_selection.guess_bonds()

        fp = plf.Fingerprint()
        fp.run(self.u.trajectory[::100], ligand_selection, pocket_selection)

        suffix = 'lig' if selection_str == '' else selection_str.replace(' ', '_')
        stem = self.pdb.split('.')[0]
        out_pkl = os.path.join(self.working_directory, 'analysis',
                               f'{stem}_fingerprint_{suffix}.pkl')
        out_csv = os.path.join(self.working_directory, 'analysis',
                               f'{stem}_fingerprint_{suffix}.csv')
        out_png = os.path.join(self.working_directory, 'analysis',
                               f'{stem}_prolig_{suffix}.png')

        fp.to_pickle(out_pkl)
        df = fp.to_dataframe()
        df.to_csv(out_csv)
        fp.plot_barcode()
        plt.savefig(out_png)
        plt.close('all')
        print(f'Fingerprint saved to {out_png}')

    def cluster_md_trajectory(self, start=0, end=0, jobname='clustering',
                               clustering_method=None, selection='name CA', n_clusters=5):
        """Cluster trajectory frames using MDAnalysis encore."""
        from MDAnalysis.coordinates.memory import MemoryReader

        savepath = os.path.join(self.working_directory, 'analysis', jobname)
        os.makedirs(savepath, exist_ok=True)

        start_frame, end_frame = self.get_frames_by_time(start_time=start, end_time=end)

        # Collect coordinates into memory (avoids ChainReader issues)
        if not hasattr(self, 'coords'):
            print('Collecting trajectory coordinates into memory…')
            coord_list = []
            for idx, ts in enumerate(self.u.trajectory[start_frame:end_frame]):
                print(f'  frame {idx}', end='\r')
                coord_list.append(ts.positions.copy())
            self.coords = np.array(coord_list)
            print(f'\nCollected {len(self.coords)} frames.')

        new_universe = mda.Universe(
            os.path.join(self.working_directory, self.pdb),
            self.coords,
            format=MemoryReader,
        )

        choice = input(
            'Atom selection for clustering:\n'
            '  (1) Pocket residues + ligand\n'
            '  (2) CA atoms (default)\n'
            '  (3) Custom selection string\n'
        )
        if choice == '1':
            selection = (
                f'(protein and (not name H*) and '
                f'(byres around 6.0 resname {self.ligand_str})) or ({self.ligand_str})'
            )
        elif choice == '3':
            selection = input('Selection string: ')
        print(f'Using selection: {selection}')

        if clustering_method is None:
            clustering_method = encore.KMeans(n_clusters=int(n_clusters))

        print('Running clustering…')
        cluster_collection = encore.cluster(
            ensembles=new_universe,
            method=clustering_method,
            select=selection,
            ncores=20,
        )
        if isinstance(cluster_collection, list):
            cluster_collection = cluster_collection[0]

        print('Saving representative structures…')
        for i, cluster in enumerate(cluster_collection.clusters):
            if len(cluster.elements) > 0:
                new_universe.trajectory[cluster.elements[0]]
                rep_file = os.path.join(savepath, f'cluster_{i+1}_rep.pdb')
                new_universe.select_atoms(f'not {self.solvent_str}').write(rep_file)

        print(f'\nClustering complete. Output: {savepath}')
        for i, cluster in enumerate(cluster_collection.clusters):
            print(f'  Cluster {i+1}: {cluster.size} frames')
        return cluster_collection

    # ------------------------------------------------------------------
    # Interactive entry point
    # ------------------------------------------------------------------

    def run(self):
        """Auto-detect topology/trajectory files and launch an interactive menu."""
        # Detect topology
        topo_files = [
            f for f in os.listdir(self.working_directory)
            if any(f.endswith(ext) for ext in ('_system.pdb', '.psf', '.prmtop', '.gro'))
        ]
        if not topo_files:
            raise FileNotFoundError(
                f'No topology file found in {self.working_directory}')
        if len(topo_files) > 1:
            for i, f in enumerate(topo_files):
                print(f'  ({i+1}) {f}')
            idx = int(input('Multiple topology files found. Pick one (number): ')) - 1
            pdbfile = topo_files[idx]
        else:
            pdbfile = topo_files[0]

        # Detect trajectory
        results_dir = os.path.join(self.working_directory, 'results')
        traj_files = [
            f for f in os.listdir(results_dir)
            if any(f.endswith(ext) for ext in ('.dcd', '.xtc', '.trj', '.nc'))
        ]
        if not traj_files:
            raise FileNotFoundError(f'No trajectory file found in {results_dir}')
        if len(traj_files) > 1:
            for i, f in enumerate(traj_files):
                print(f'  ({i+1}) {f}')
            idx = int(input('Multiple trajectory files found. Pick one (number): ')) - 1
            dcdfile = traj_files[idx]
        else:
            dcdfile = traj_files[0]

        print(f'Topology : {pdbfile}')
        print(f'Trajectory: {dcdfile}')
        apo = input('Apo protein? (1) Yes  (2) No\n') == '1'
        self.read_pdb_dcd(pdbfile, dcdfile, apo=apo)

        menu = (
            '\nSelect analysis:\n'
            '  (1) RMSD\n'
            '  (2) Extract frames\n'
            '  (3) Distance analysis\n'
            '  (4) RMSF\n'
            '  (5) Protein-ligand fingerprint\n'
            '  (6) Dihedral angle\n'
            '  (7) Pairwise RMSD heatmap\n'
            '  (8) Secondary structure (DSSP)\n'
            '  (9) Clustering\n'
            '  (0) Quit\n'
        )
        while True:
            job = input(menu)
            if job == '1':
                self.rmsd()
            elif job == '2':
                s, e, n = input('start (time), end (time), n_frames: ').split(',')
                self.extract_complex(start=float(s), end=float(e), slices=int(n))
            elif job == '3':
                self.distance_analysis()
            elif job == '4':
                self.rmsf()
            elif job == '5':
                lig_str = input('Default ligand (Y) or custom selection string? ')
                if lig_str.upper() == 'Y':
                    self.pro_lig_int()
                else:
                    flag = input('Is this a ligand (1) or a protein segment (2)? ')
                    self.pro_lig_int(selection_str=lig_str, lig=(flag == '1'))
            elif job == '6':
                lig = input('Ligand dihedral (Y) or protein dihedral (N)? ').upper() == 'Y'
                self.dihedral_angle(ligand=lig)
            elif job == '7':
                self.heatmap()
            elif job == '8':
                self.DSSP()
            elif job == '9':
                s, e, n = input('start (time), end (time), n_clusters: ').split(',')
                self.cluster_md_trajectory(start=float(s), end=float(e),
                                           n_clusters=int(n))
            elif job == '0':
                break


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    work_dir = input('Working directory name (under working_directory/): ')
    analyzer = MDAnalyzer(directory=work_dir)
    analyzer.run()
