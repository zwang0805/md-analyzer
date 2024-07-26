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
from sklearn.cluster import AgglomerativeClustering

class md_analysis():
    def __init__(self, directory=''):
        self.path = os.path.split(os.path.realpath(__file__))[0]
        if directory == '':
            self.working_directory = os.path.join(self.path, 'working_directory')
        else:
            self.working_directory = os.path.join(self.path, 'working_directory', directory)
        if not os.path.exists('%s/analysis' % self.working_directory):
            os.mkdir('%s/analysis' % self.working_directory)
        self.solvent_str='(resname WAT or resname SOL or resname HOH)'
        self.ion_str='(resname NA or resname CL or resname NA? or resname CL?)'
        self.ligand_str= '(not (%s or %s or protein))' % (self.solvent_str, self.ion_str)
        
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
        from MDAnalysis.topology.guessers import guess_types
        self.u = mda.Universe(os.path.join(self.working_directory, pdb), os.path.join(self.working_directory,'results', dcd))
        self.dt=self.u.trajectory.dt

        if '.gro' in pdb:
            self.u.atoms.guess_bonds()
            elements = guess_types(self.u.atoms.types)
            self.u.add_TopologyAttr('elements', elements)
        self.pdb=pdb
        self.apo=apo
        
        prot = self.u.select_atoms('protein')
        ref_u=self.u.copy()
        ref_u.trajectory[0]
        reference = ref_u.select_atoms('protein')
        ag = self.u.select_atoms('not (%s or %s)' % (self.ion_str, self.solvent_str))
        if apo:
            ag.guess_bonds()
        water=self.u.select_atoms(self.solvent_str)
        self.workflow = (
                    mda.transformations.center_in_box(prot, center='mass'),
                    mda.transformations.wrap(ag, compound='fragments'),
                    mda.transformations.wrap(water, compound='residues'),
                    mda.transformations.fit_rot_trans(prot, reference)
                    )

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
        atom_label2=atom_label.replace('ligand',self.ligand_str)
        atom1_exp,atom2_exp=atom_label2.split(',')
        atom1 = self.u.select_atoms(atom1_exp)
        atom2 = self.u.select_atoms(atom2_exp)
        ligand_atom=self.u.select_atoms(self.ligand_str)
        
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
        plt.clf()
        plt.close('all')
        print('the result figure is saved in %s/analysis/%s_dist_analysis_%s.png\n' % (self.working_directory, self.pdb,suffix))

    def heatmap(self):
        dist_matrix_protein = self.RMSD_dist_frames(self.u, "protein")
        print(dist_matrix_protein)
        dist_matrix_ligand = self.RMSD_dist_frames(self.u, self.ligand_str)
        max_dist = max(np.amax(dist_matrix_ligand), np.amax(dist_matrix_protein))
        fig, ax = plt.subplots(1, 2)
        fig.suptitle("RMSD between the frames")

        # protein image
        img1 = ax[0].imshow(dist_matrix_protein, cmap="viridis", vmin=0, vmax=max_dist)
        ax[0].title.set_text("protein")
        ax[0].set_xlabel("frames")
        ax[0].set_ylabel("frames")

        # ligand image
        img2 = ax[1].imshow(dist_matrix_ligand, cmap="viridis", vmin=0, vmax=max_dist)
        ax[1].title.set_text("Ligand")
        ax[1].set_xlabel("frames")

        fig.colorbar(img1, ax=ax, orientation="horizontal", fraction=0.1, label="RMSD (Å)")
        fig.savefig(os.path.join(self.working_directory, 'analysis', '%s_rmsd2.png' % self.pdb))
        plt.close()

    def RMSD_dist_frames(self,universe, selection):
        """Calculate the RMSD between all frames in a matrix.
    
        Parameters
        ----------
        universe: MDAnalysis.core.universe.Universe
            MDAnalysis universe.
        selection: str
            Selection string for the atomgroup to be investigated, also used during alignment.
    
        Returns
        -------
        array: np.ndarray
            Numpy array of RMSD values.
        """
        pairwise_rmsd = diffusionmap.DistanceMatrix(universe, select=selection)
        pairwise_rmsd.run()
        return pairwise_rmsd.results.dist_matrix
        
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
                               index=rmsd_analysis.rmsd[:, 0]*self.dt)
        else:
            rmsd_analysis = rms.RMSD(self.u, select='backbone', groupselections=['name CA', 'protein', self.ligand_str])
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
        plt.close('all')
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

        pro = universe.universe.select_atoms('not %s' % self.ion_str)
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
        #average = align.AverageStructure(self.u, self.u, select=selection,
                                         #ref_frame=0).run()
        #ref = average.results.universe
        #aligner = align.AlignTraj(self.u, ref, select=selection,
        #                          in_memory=True).run()
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
        lig=self.rmsf_basis(self.ligand_str)
        print(lig[0], lig[1])
        plt.plot(lig[0], lig[1],"ok-")
        plt.xlabel('Atom index')
        plt.ylabel('RMSF ($\AA$)')
        plt.savefig(os.path.join(self.working_directory,'analysis','%s_ligand_rmsf.png' % (self.pdb)))
        plt.close('all')
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

        Returns
        -------
        1. plot of the protein-ligand interaction
        2. the fingerprint csv file that facilitate you to calculate the overall percent of precence of any interaction of interest.
        """

        ligand_selection=self.u.select_atoms(self.ligand_str)
        if '.pdb' in self.pdb:
            ligand_selection.guess_bonds(vdwradii={"Cl": 1.75,"Br": 1.85,"I": 1.98})
        ligand_mol = plf.Molecule.from_mda(ligand_selection)
        protein_selection = self.u.select_atoms("protein")
        if '.pdb' in self.pdb:
            protein_selection.guess_bonds()
        pocket_selection = self.u.select_atoms("(protein) and byres around 6.0 group ligand",ligand=ligand_selection)
        if '.pdb' in self.pdb:
            pocket_selection.guess_bonds()

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
        plt.close('all')
        print('the result figure is saved in %s/analysis/%s_prolig.png\n' % (self.working_directory, self.pdb))
    
    def conserved_water(self,threshold=0.5,selection='byres (around 3.5 protein)'):
        pdbfile = input('please specify the name of the pdb file:\n')
        print('read pdb file...')
        u0 = mda.Universe(os.path.join(self.working_directory, 'analysis', pdbfile))
        print('iterating trajectory...')
        
        coords_list=[]
        water_list=[]
        for idx,ts in enumerate(u0.trajectory):
            print(idx)
            water_atoms = u0.select_atoms(selection)
            for atom in water_atoms:
                #print(atom.name,atom.index,atom.resid,atom.position)
                if atom.name=='O':
                    full_name='%s_%s' % (idx,atom.resid)
                    coords_list.append(atom.position)
                    water_list.append(full_name)
                    print(full_name,atom.position)
                    
        coords_ary=np.array(coords_list)
        ac = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='single',
                                     distance_threshold=0.5)
        labels = ac.fit_predict(coords_ary)
        label_dict={}
        for label in labels:
            if not label in label_dict.keys():
                label_dict[label]=0
            label_dict[label] += 1
        
        if not os.path.exists(os.path.join(self.working_directory,'analysis', 'conserved_water')):
            os.mkdir('%s/analysis/conserved_water' % (self.working_directory))
        csv_file=os.path.join(self.working_directory,'analysis','conserved_water', '%s_conserved_waters.csv' % pdbfile)
        with open(csv_file,'a') as f:
            for label,cluster in zip(labels,water_list):
                f.write('%s,%s,%s,%.3f\n' %(label,cluster,label_dict[label],label_dict[label]/len(u0.trajectory)))
        df=pd.read_csv(csv_file, header=None)
        df.columns=['cluster_id', 'water_id', 'entries', 'frequency']
        df.to_csv(csv_file)
        filtered_df = df[df['frequency'] > threshold]
        save_dict={}
        if not filtered_df.empty:
            for each_id in filtered_df['water_id'].values:
                prot_id,wat_id =each_id.split('_')
                if not prot_id in save_dict.keys():
                    save_dict[prot_id]=[]
                save_dict[prot_id].append(wat_id)
        for idx in save_dict.keys():
            a=' or resid '.join(save_dict[idx])
            print(save_dict[idx])
            ts=u0.trajectory[int(idx)]
            protein_to_save = u0.select_atoms('protein or %s or resid %s' % (self.ligand_str,a))
            
            protein_to_save.write(os.path.join(self.working_directory,'analysis','conserved_water','%s_%s.pdb' % (pdbfile,idx)))
        
    def run(self):
        for file_name in os.listdir(self.working_directory):
            if "_system.pdb" in file_name:
                pdbfile=file_name
        for file_name in os.listdir(os.path.join(self.working_directory,'results')):
            if "_trajectory.dcd" in file_name:
                dcdfile=file_name
        print('detected pdbfile and dcffile as %s and %s' %(pdbfile,dcdfile))
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
            elif job == '6':
                print('analyzing protein-ligand heatmap...')
                self.heatmap()
            elif job == '7':
                print('analyzing conserved water molecules...')
                self.conserved_water()
            elif job == '0':
                break
def test():
    work_dir='test'
    mda_ana=md_analysis(directory=work_dir)
    #mda_ana.read_pdb_dcd('5o1c_14586_system.pdb', '5o1c_14586_2ns_new_trajectory.dcd',apo=False)
    mda_ana.conserved_water()

if __name__ == '__main__':
    work_dir=input('please input the name of working directory\n')
    mda_ana=md_analysis(directory=work_dir)
    mda_ana.run()


