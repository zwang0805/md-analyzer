import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import MDAnalysis as mda
import pandas as pd
import numpy as np
from MDAnalysis.analysis import rms
import matplotlib.pyplot as plt
from MDAnalysis.analysis import align, diffusionmap
from MDAnalysis import transformations
from MDAnalysis.analysis import distances
import prolif as plf

class md_analysis():
    def __init__(self, directory=''):
        self.path = os.path.split(os.path.realpath(__file__))[0]
        if directory == '':
            self.working_directory = os.path.join(self.path)
        else:
            self.working_directory = directory
        if not os.path.exists('%s/analysis' % self.working_directory):
            os.mkdir('%s/analysis' % self.working_directory)
        self.solvent_str='WAT'
        

    def read_pdb_dcd(self,pdb,dcd,apo=False):
        """Read the input topology and trajectory file
        Parameters
        ----------
        pdb: str,
        name of the topology file, any topology file that can be recognized by MDAnalysis, including: .pdb, .prmtop, .gro, .psf, .xyz, etc.
        dcd: str,
        name of the trajectory file, any trajectory file can be recognized by MDAnalysis, including: .dcd, .trj, .xtc, etc.
        apo: boolean,
        The MDAnalysis will only consider protein structure when correcting the coordinates of structure based on Periodic Boundary Conditions. 
        For a protein structure, no additional trasformation will be performed. For a protein-ligand complex structure, the ligand will not 
        be transformed based on Periodic Boundary Conditions by default, so additional on-the-fly transformations are added to address 
        this issue with an incresed of analysis time. This script does not ensure the successful pdb correction for every system, please check
        the result carefully.
        """
        self.u = mda.Universe(os.path.join(self.working_directory, pdb), os.path.join(self.working_directory, dcd))
        self.dt=self.u.trajectory.dt
        if '.pdb' in pdb:
            self.solvent_str='HOH'
        elif '.gro' in pdb:
            from MDAnalysis.topology.guessers import guess_types
            self.solvent_str='SOL'
            self.u.atoms.guess_bonds()
            elements = guess_types(self.u.atoms.types)
            self.u.add_TopologyAttr('elements', elements)
        self.pdb=pdb
        self.apo=apo
        if apo:
            return

        
        prot = self.u.select_atoms('protein')
        ref_u=self.u.copy()
        reference = ref_u.select_atoms('protein')
        ag = self.u.select_atoms('not (resname NA or resname CL)')
        
        self.workflow = (mda.transformations.unwrap(ag),
                    mda.transformations.center_in_box(prot, center='mass'),
                    mda.transformations.wrap(ag, compound='fragments'),
                    mda.transformations.fit_rot_trans(prot, reference))

        self.u.trajectory.add_transformations(*self.workflow)
        
    
    def distance_analysis(self,atom_label='',dist_step=100):
        """Calculate the distance between two atoms of the MD results.

        Parameters
        ----------
        jobname: str,
        name that will be used in saving the output files.
        atom_label: str,
        string expression of the two atoms of interest, seperated by "," (e.g. resid 30 and name CZ,resid 163 and name CA)
        dist_step: int,
        to accelerate the distance analysis, the distance between the analyzed atoms can be only calculated every "dist_step" frames.
        use 1 to perform distance calculation for each frame.

        Notice
        -------
        this script is written to analyze the output files of OpenMM, therefore the resname of water is used as HOH, and Na and Cl are used
        as the default salts. If you are using this script to analyze the results outputed by other softwares, please change the "ligand_exp"
        according to your softwares and protocols

        Output
        -------
        1. the plot of distance vs. time (ps) in png format in the analysis folder.
        2. the file of distance vs. time (ps) in csv format.

        """
        if atom_label=='':
            atom_label=input('please input atom label, you can use ligand to indicate your small molecule (e.g. resid 30 and name CZ, ligand and name S25):\nplease refer to the following website for atom label edition:\nhttps://userguide.mdanalysis.org/stable/selections.html\n')
        ligand_exp = '(not (resname %s or resname NA or resname CL or protein))' % self.solvent_str
        atom_label2=atom_label.replace('ligand',ligand_exp)
        atom1_exp,atom2_exp=atom_label2.split(',')
        atom1 = self.u.select_atoms(atom1_exp)
        atom2 = self.u.select_atoms(atom2_exp)
        ligand_atom=self.u.select_atoms(ligand_exp)
        
        dist = []
        time_ids=[]
        for idx, ts in enumerate(self.u.trajectory[::dist_step]):
            dist_arr = distances.distance_array(atom1.positions, atom2.positions, box=self.u.dimensions)
            dist.append(dist_arr[0,0])
            
            time_ids.append(ts.time)
            print(int(ts.time), dist_arr[0,0])

        ary=np.array([time_ids, dist])
        df=pd.DataFrame(ary.T,columns=['time (ps)','distance (A)'])
        df.to_csv(os.path.join(self.working_directory,'analysis','%s_distance.csv' % self.pdb))
        # Create a line plot using the values in dist and idx
        plt.plot(time_ids, dist)
        
        # Add labels and title
        plt.xlabel('Time (ps)')
        plt.ylabel('Distance ($\AA$)')
        plt.title('Distance vs. Time (mean,std = %.2f,%.2f)' % (np.mean(np.array(dist)),np.std(np.array(dist))))
        
        # Save the figure to a file
        suffix=atom_label.replace(' ','_').replace(',','_')
        plt.savefig(os.path.join(self.working_directory,'analysis','%s_dist_analysis_%s.png' % (self.pdb,suffix)), dpi=300)
        plt.close()
        print('the result figure is saved in %s/analysis/%s_dist_analysis_%s.png\n' % (self.working_directory, self.pdb,suffix))

    def rmsd(self,rmsd_step=100):
        """Calculate the RMSD of the MD trajectory.

        Parameters
        ----------

        rmsd_step: int,
        to accelerate the distance analysis, the distance between the analyzed atoms can be only calculated every "rmsd_step" frames.
        use 1 to perform distance calculation for each frame.

        Output
        -------
        1. the plot of rmsd vs. time (ps) in png format in the analysis folder.
        2. the file of rmsd vs. time (ps) in csv format.
        
        """

        if self.apo:
            rmsd_analysis = rms.RMSD(self.u, select='backbone', groupselections=['name CA', 'protein'])
            rmsd_analysis.run(step=rmsd_step)
            rmsd_df = pd.DataFrame(rmsd_analysis.rmsd[:, 2:],
                               columns=['backbone','C-alphas', 'protein'],
                               index=rmsd_analysis.rmsd[:, 0])
        else:
            rmsd_analysis = rms.RMSD(self.u, select='backbone', groupselections=['name CA', 'protein', 'not (resname %s or resname HW or resname OW or resname SOL or resname NA or resname CL or protein)' % self.solvent_str])
            rmsd_analysis.run(step=rmsd_step)
            rmsd_df = pd.DataFrame(rmsd_analysis.rmsd[:, 2:],
                               columns=['backbone','C-alphas', 'protein','ligand'],
                               index=rmsd_analysis.rmsd[:, 0]*self.dt)
        rmsd_df.index.name = 'Time (ps)'
        rmsd_df.head()
        print(rmsd_df)
        ax=rmsd_df.plot(title='RMSD')
        fig=ax.get_figure()
        fig.savefig(os.path.join(self.working_directory,'analysis','%s_rmsd.png' % self.pdb))
        rmsd_df.to_csv(os.path.join(self.working_directory,'analysis','%s_rmsd.csv' % self.pdb))
        plt.close()
        print('the result figure is saved in %s/analysis/%s_rmsd.png\n' % (self.working_directory, self.pdb))

    def extract_complex(self,start=0, end=0,slices=50):
        """Extract frames from the MD trajectory within the specified time range and save as PDB files.

        Parameters
        ----------
        start: int,
        the time (ns) to start recording the structure

        end: int,
        the time (ns) to stop recording the structure

        slices: int,
        number of structures to output

        Output
        -------
        a pdb file with multiple structures
        """
        
        universe=self.u
        total_frames=len(universe.trajectory)
        selected_frames=[]
        time_list=[]
        for ts in range(total_frames):
            time=(ts+1)*self.dt/1000
            if end==0:
                if time >= start:
                    selected_frames.append(ts)
                    time_list.append(time)
            else:
                if time>=start and time<=end:
                    selected_frames.append(ts)
                    time_list.append(time)

        pro = universe.universe.select_atoms('not (resname NA or resname CL)')
        num_frames=len(selected_frames)
        
        if num_frames<slices:
            print(time_list)
            pro.write(os.path.join(self.working_directory,'analysis','%s_frames_%s.pdb' % (self.pdb,slices)), frames=selected_frames)
        else:
            print(time_list[::int(num_frames/slices)])
            pro.write(os.path.join(self.working_directory,'analysis','%s_frames_%s.pdb' % (self.pdb,slices)), frames=selected_frames[::int(num_frames/slices)])
        print('the result frames are saved in %s/analysis/%s_frames_(time).pdb\n' % (self.working_directory, self.pdb))
        
    def rmsf_basis(self,selection):
        """Calculate the RMSF of selected atomgroup in a MD simulation.

        Parameters
        ----------
        selection: str
        Selection string for the atomgroup to be investigated, also used during alignment.

        Returns
        -------
        sequence of atom index and its corresponding rmsf

        """
        average = align.AverageStructure(self.u, self.u, select=selection,
                                         ref_frame=0).run()
        ref = average.results.universe
        aligner = align.AlignTraj(self.u, ref, select=selection,
                                  in_memory=True).run()
        ag = self.u.select_atoms(selection)
        R = rms.RMSF(ag).run()
        atoms_num=len(ag)
        return range(1,atoms_num+1), R.results.rmsf[0:atoms_num]
    
    def ligand_rmsf(self):
        """Calculate the RMSF of ligands in a MD simulation.

        Output
        -------
        the plot of atom index and rmsf
        
        """
        lig=self.rmsf_basis('not (protein or resname %s or resname NA or resname CL)' % self.solvent_str)
        print(lig[0], lig[1])
        plt.plot(lig[0], lig[1],"ok-")
        plt.xlabel('Atom index')
        plt.ylabel('RMSF ($\AA$)')
        plt.savefig(os.path.join(self.working_directory,'analysis','%s_ligand_rmsf.png' % (self.pdb)))
        plt.close()
        print('the result figure is saved in %s/analysis/%s_ligand_rmsf.png\n' % (self.working_directory, self.pdb))
    
    def pro_lig_int(self):
        """Calculate the protein-ligand interaction fingerprint using ProLif package.

        Notice
        -------
        1. If you use ProLIF in your research, please cite the following paper:

        Bouysset, C., Fiorucci, S. ProLIF: a library to encode molecular interactions as fingerprints.
        J Cheminform 13, 72 (2021). https://doi.org/10.1186/s13321-021-00548-6

        2. if you’re loading a structure from a file that doesn’t explicitely contain bond orders and formal charges, such as a PDB file or most MD trajectory files.
        MDAnalysis will infer those from the atoms connectivity, which requires all atoms including hydrogens to be present in the input file. In some cases,
        some atomic clashes may be incorrectly classified as bonds and will prevent the conversion of MDAnalysis molecules to RDKit.
        Since MDAnalysis uses van der Waals radii for bond detection, you can add the default radii if there is a atom is not recognized.

        ligand_selection.guess_bonds(vdwradii={"Cl": 1.75,"Br": 1.85,"I": 1.98})
        
        3. If errors are showing and its related to multiprocess, please run the following command before you run this python script:
        export OPENBLAS_NUM_THREADS=1

        Returns
        -------
        1. plot of the protein-ligand interaction
        2. the fingerprint csv file that facilitate you to calculate the overall percent of precence of any interaction of interest.
        """
        
        ligand_selection=self.u.select_atoms('not (protein or resname %s or resname NA or resname CL)' % self.solvent_str)
        if '.pdb' in self.pdb:
            ligand_selection.guess_bonds(vdwradii={"Cl": 1.75,"Br": 1.85,"I": 1.98})
        ligand_mol = plf.Molecule.from_mda(ligand_selection)
        protein_selection = self.u.select_atoms("protein")
        if '.pdb' in self.pdb:
            protein_selection.guess_bonds()
        pocket_selection = self.u.select_atoms("(protein) and byres around 6.0 group ligand",ligand=ligand_selection)
        if '.pdb' in self.pdb:
            pocket_selection.guess_bonds()
        #print(plf.Fingerprint.list_available())
        # use default interactions
        fp = plf.Fingerprint()
        
        # run on a slice of the trajectory frames: from begining to end with a step of 10
        fp.run(self.u.trajectory[::10], ligand_selection, pocket_selection)
        #save the results
        fp.to_pickle(os.path.join(self.working_directory,'analysis','%s_fingerprint.pkl' % (self.pdb)))
        df = fp.to_dataframe()
        print(df)
        df.to_csv(os.path.join(self.working_directory,'analysis','%s_fingerprint.csv' % (self.pdb)))
        fp.plot_barcode()
        plt.savefig(os.path.join(self.working_directory,'analysis','%s_prolig.png' % (self.pdb)))
        plt.close()
        print('the result figure is saved in %s/analysis/%s_prolig.png\n' % (self.working_directory, self.pdb))

    def run(self):
        pdbfile, dcdfile = input(
            'please specify the name of the pdb file and dcd file e.g. com.prmtop, traj.dcd\n').split(',')
        if input('apo structure? (1) Yes (2) No\n') == '1':
            apo = True
        else:
            apo = False
        print('reading', pdbfile, 'and', dcdfile)
        self.read_pdb_dcd(pdbfile, dcdfile,apo=apo)
        while True:
            job = input(
                'please select the mode:\n(1) rmsd\n(2) extract complex\n(3) distance analysis\n(4) ligand rmsf\n(5) protein-ligand fingerprint\n(0) quit\n')
            if job == '1':
                self.rmsd()
            elif job == '2':
                st, ed, num_frames = input(
                    'please input: starting time (ns),ending time(ns), number of frames\n').split(',')
                self.extract_complex(start=float(st), end=float(ed), slices=int(num_frames))
            elif job == '3':
                self.distance_analysis()
            elif job == '4':
                print('running ligand atom rmsf...')
                self.ligand_rmsf()
            elif job == '5':
                print('analyzing protein-ligand interaction...')
                self.pro_lig_int()
            elif job == '0':
                break


if __name__ == '__main__':
    work_dir=input('please input the name of working directory\n')
    mda_ana=md_analysis(directory=work_dir)
    mda_ana.run()


