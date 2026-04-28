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
from sklearn.cluster import AgglomerativeClustering
from MDAnalysis.analysis import gnm
from MDAnalysis.transformations.base import TransformationBase
import mdaencore as encore

# 定义 GroupHug 转换类
class GroupHug(TransformationBase):
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
        self.ion_str='(resname NA or resname CL or resname NA? or resname CL? or resname MG or resname MG* or resname MN*)'
        self.ligand_str= '(not (%s or %s or protein))' % (self.solvent_str, self.ion_str)
        self.delete_units=''

    def universe_info(self):
        # 查看所有可用的 segid
        print(set(self.u.segments.segids))
        # 查看所有可用的 chainID
        #print(set(self.u.atoms.chainIDs))
        pro_list=[]
        pro_residues = self.u.select_atoms('protein')
        for residue in pro_residues:
            pro_list.append(residue.resname)
        unique_pro_list=set(pro_list)
        nonpro_list=[]
        nonpro_residues = self.u.select_atoms('not protein')
        for residue in nonpro_residues:
            nonpro_list.append(residue.resname)
        unique_nonpro_list=set(nonpro_list)
        residue_list=[]
        for residue in self.u.residues:
            residue_list.append(residue.resname)
        unique_residue_list=set(residue_list)
        print('total residues:',unique_residue_list)
        print('protein residues:',unique_pro_list)
        print('other residues:',unique_nonpro_list)

    def write_prmtop(self,st):
        import parmed as pmd
        # 首先将结构保存为中间格式（PDB）
        self.u.atoms.write('temp.pdb')
        # 使用ParmEd加载中间文件
        structure = pmd.load_file('temp.pdb')
        # 设置必要的力场参数（这步很关键，因为PDB不包含力场信息）
        # 下面是一个简单示例，实际使用需要更完整的参数设置
        for atom in structure.atoms:
            atom.atomic_number = atom.element_atomic_number # 设置原子序数
            # 设置其他必要参数...
        # 保存为prmtop文件
        structure.save('output.prmtop', overwrite=True)
        structure = pmd.openmm.load_topology(self.topology, self.system, xyz=self.positions)
        jobname=self.pdbname.split('.')[0]
        savepath=os.path.join(self.working_directory,jobname,'results')
        structure.save(os.path.join(savepath,'%s_system.prmtop' % jobname), overwrite=True)
        structure.save(os.path.join(savepath,'%s_system.inpcrd' % jobname), overwrite=True)

    def get_frames_by_time(self, start_time, end_time):
        """
        根据输入的时间范围（皮秒），计算并返回对应的轨迹帧编号。
        参数:
            self.u: MDAnalysis.Universe 对象
            start_time (float/int): 起始时间 (ps)
            end_time (float/int): 结束时间 (ps)。如果传入 -1，则表示一直到轨迹末尾。
        返回:
            numpy.ndarray: 包含符合条件的帧编号（整数索引）的一维数组。
        """
        # 1. 提取轨迹的时间基准信息
        dt = self.dt
        t0 = self.u.trajectory[0].time # 轨迹第一帧的真实物理时间
        total_frames = len(self.u.trajectory)

        # 2. 计算起始帧索引
        # 公式： (目标时间 - 起始时间) / 时间步长
        start_idx = int(round((float(start_time) - t0) / dt))
        start_idx = max(0, start_idx) # 防止传入的 start_time 过小导致负数索引

        # 3. 计算结束帧索引
        if end_time == -1:
            end_idx = total_frames - 1
        else:
            end_idx = int(round((float(end_time) - t0) / dt))
            end_idx = min(total_frames - 1, end_idx) # 防止传入的 end_time 过大导致索引越界

        # 4. 异常处理：检查范围是否合法
        if start_idx > end_idx:
            print(f"警告: 计算得到的起始帧({start_idx})大于结束帧({end_idx})。请检查输入的时间范围。")
            return np.array([], dtype=int)

        # 5. 生成连续的帧编号序列
        # arange 的区间是左闭右开的，所以结束值需要 +1
        frames = (start_idx, end_idx + 1)
        print(f'current dt is {dt}, and total frames is {total_frames}, and total simulation time is {float(dt*total_frames)}\n')
        print(f'the starting time is {start_time}, and the corresponding frame is {start_idx}\nthe ending time is {end_time}, and the corresponding frame is {end_idx}\n')
        return frames

    def cluster_md_trajectory(self, start=0, end=0, jobname='clustering', clustering_method=None, selection='name CA', n_clusters=5):
        from MDAnalysis.coordinates.memory import MemoryReader
        # 1. 路径准备
        savepath = os.path.join(self.working_directory, 'analysis', jobname)
        os.makedirs(savepath, exist_ok=True)

        # 2. 计算提取帧数
        start_frame,end_frame=self.get_frames_by_time(start_time=start,end_time=end)

        # 3. 【关键优化】批量提取坐标到内存
        # 使用列表推导式比循环 append 快得多，且能避免 ChainReader 报错
        sub_traj = self.u.trajectory[start_frame : end_frame + 1]
        print("collecting coordinates...")
        if not hasattr(self, 'coords'):
            coord_list=[]
            for idx,ts in enumerate(sub_traj):
                print(f'collecting the coordinates of {idx}th frame...')
                coord_list.append(ts.positions)
            coords = np.array(coord_list)
            self.coords=coords
        # 4. 创建基于内存的新 Universe
        # 这样 encore 处理的是纯内存数据，不会再触发文件读取器的 Bug
        new_universe = mda.Universe(os.path.join(self.working_directory, self.pdb), 
                                     self.coords, 
                                     format=MemoryReader)

        # 5. 交互式选择逻辑修复
        selection_query = input('当前默认选择为 CA 原子，是否聚焦于 Pocket 残基？\n(1) 是 \n(2) 默认(CA) \n(3) 手动输入选择语句\n')
        if selection_query == '1':
            selection = f'(protein and (not name H*) and (byres around 6.0 resname {self.ligand_str})) or ({self.ligand_str})'
        elif selection_query == '3':
            selection = input('请输入您的选择语句 (例如: "resname LIG or protein"): ')

        print(f'当前选择语句: {selection}')

        # 6. 设置聚类方法
        if clustering_method is None:
            clustering_method = encore.KMeans(n_clusters=int(n_clusters))

        # 7. 执行聚类 (ncores=20 保持不变)
        print("正在进行聚类计算...")
        cluster_collection = encore.cluster(
            ensembles=new_universe,
            method=clustering_method,
            select=selection,
            ncores=20
        )

        if isinstance(cluster_collection, list):
            cluster_collection = cluster_collection[0]

        # 8. 保存聚类代表构象
        print("正在提取代表构象并保存...")
        for i, cluster in enumerate(cluster_collection.clusters):
            if len(cluster.elements) > 0:
                # 注意：这里的 cluster.elements 是相对于 new_universe (内存) 的索引
                # 这里的坐标提取非常快
                new_universe.trajectory[cluster.elements[0]] 
                rep_file = os.path.join(savepath, f"cluster_{i+1}_representative.pdb")

                # 排除溶剂并保存
                atoms_to_save = new_universe.select_atoms(f'not {self.solvent_str}')
                atoms_to_save.write(rep_file)

        print(f"\n=== 聚类完成 ===")
        print(f"输出目录: {savepath}")
        for i, cluster in enumerate(cluster_collection.clusters):
            print(f"簇 {i+1}: 包含 {cluster.size} 帧")

        return cluster_collection

    def DSSP(self,residues='all',dssp_step=50):
        #pro_residues = self.u.select_atoms('protein')
        from MDAnalysis.analysis.dssp import translate, DSSP
        start=0
        end=-1
        if '-' in residues:
            start,end=residues.split('-')
            start=int(start)
            end=int(end)
        run = DSSP(self.u).run(step=dssp_step)
        if end==-1:
            secondary_structure_ary = run.results.dssp[:, start:]
        else:
            secondary_structure_ary = run.results.dssp[:, start:end]
        pdbname=self.pdb.split('.')[0]
        df = pd.DataFrame(secondary_structure_ary)
        residue_numbers = run.results.resids
        print(residue_numbers)
        df.columns=residue_numbers
        df.to_csv(os.path.join(self.working_directory,'analysis','%s_dssp.csv' % (pdbname)))
        print('''loop '-' (index 0), helix 'H' (index 1), and sheet 'E' (index 2)\n ''')

    def remove_subunits(self,delete_units='',solvent_selection='',ligand_selection=''):
        self.universe_info()
        if delete_units=='':
            delete_units=input('Do you want to delete some units?\n(1)No\n(2)input selection e.g. chainID B\nplease refer to the following website for atom label edition:\nhttps://userguide.mdanalysis.org/stable/selections.html\n')
        if delete_units!='1':
            self.delete_units=delete_units
            selection = self.u.select_atoms('not (%s)' % delete_units)
            #selection.atoms.elements = element_list
            new_pdb= self.pdb.split('.')[0] + '_no_%s' % delete_units.replace(' ','_') + '_system.pdb'
            self.pdb=new_pdb
            new_dcd= self.dcd.split('.')[0] + '_no_%s.' % delete_units.replace(' ','_') + self.dcd.split('.')[-1]
            self.dcd=new_dcd
            # 保存新文件
            print(self.pdb,self.dcd,'\n')
            if 'psf' in self.pdb:
                from MDAnalysis.coordinates.PSF import PSFWriter
                with PSFWriter(os.path.join(self.working_directory,self.pdb)) as psf:
                    psf.write(selection)
            else:
                selection.write(os.path.join(self.working_directory,self.pdb))
            with mda.Writer(os.path.join(self.working_directory,'results', self.dcd), selection.n_atoms) as W:
                for ts in self.u.trajectory:
                    W.write(selection)
            self.u = mda.Universe(os.path.join(self.working_directory, self.pdb), os.path.join(self.working_directory,'results', self.dcd))
            print(f'open new file: {self.pdb}, {self.dcd}...')
            #from MDAnalysis.topology.guessers import guess_types
            #guessed_elements = guess_types(self.u.atoms.names)
            #self.u.add_TopologyAttr('elements', guessed_elements)
            self.universe_info()

        if solvent_selection!='no':
            if solvent_selection=='':
                solvent_selection=input('current solvent is defined as: %s, do you want to specify solvent selection?\n(1)No\n(2)input solvent resname e.g. TIP3,SOL\n' % self.solvent_str)
            if not solvent_selection=='1':
                self.solvent_str='(resname '
                for solv_str in solvent_selection.split(','):
                    solv_str=solv_str.strip()
                    self.solvent_str+='%s or resname ' % solv_str
                self.solvent_str=self.solvent_str[0:-12]+')'
                self.ligand_str= '(not (%s or %s or protein))' % (self.solvent_str, self.ion_str)
                print('Current solvent is defined as: %s' % self.solvent_str)

        if not self.apo:
            if ligand_selection!='no':
                if ligand_selection=='':
                    ligand_selection=input('current ligand is defined as: %s, do you want to specify ligand selection?\n(1)No\n(2)input ligand resname e.g. UNL\n' % self.ligand_str)
                if not ligand_selection=='1':
                    self.ligand_str='(resname '
                    for lig_str in ligand_selection.split(','):
                        lig_str=lig_str.strip()
                        self.ligand_str+='%s or resname ' % lig_str
                    self.ligand_str=self.ligand_str[0:-12]+')'
                    print('Current ligand is defined as: %s' % self.ligand_str)

    def read_pdb_dcd(self,pdb,dcd,apo=False,remove_subunits='',solv_selection='',lig_selection='',align_chain='',dt=''):
        """Read the input topology and trajectory file
        Parameters
        ----------
        pdb: str,
            name of the topology file, any topology file that can be recognized by MDAnalysis, including: .pdb, .prmtop, .gro, .psf, .xyz, etc.
        dcd: str,
            name of the trajectory file, any trajectory file can be recognized by MDAnalysis, including: .dcd, .trj, .xtc, etc.
        apo: boolean,
            The MDAnalysis will only consider protein structure when correcting the coordinates of structure based on Periodic Boundary Conditions. 
            For a protein structure, no additional trasformation will be performed. For a
            protein-ligand complex structure, the ligand will not be transformed based on Periodic Boundary Conditions by default, so additional on-the-fly transformations are added to address 
            this issue with an incresed of analysis time. This script does not ensure the successful pdb correction for every system, please check
            the result carefully.
        """
        from MDAnalysis.topology.guessers import guess_types
        self.u = mda.Universe(os.path.join(self.working_directory, pdb), os.path.join(self.working_directory,'results', dcd))
        wat= self.u.select_atoms('resname HOH and resid 1')
        if len(wat)>3:
            print('water molecule 1 seems contain too much atoms:', len(wat))
            print('please delete this water molecule: resname HOH and resid 1')

        # 打印前几个时间点来推断单位
        for i, ts in enumerate(self.u.trajectory):
            print(f"Frame {i}: time = {ts.time}")
        self.dt=self.u.trajectory.dt
        if dt=='':
            dt_query=input(f'the time step is {self.dt}, do you want to contine(Y)? or you can change the time step in (ns) to:\n')
            if dt_query!='Y':
                self.dt=float(dt_query)
        self.pdb=pdb
        self.apo=apo
        self.dcd=dcd
        if remove_subunits!='no':
            self.remove_subunits(delete_units=remove_subunits,solvent_selection=solv_selection,ligand_selection=lig_selection)
        if '.gro' in pdb:
            self.u.atoms.guess_bonds()
            elements = guess_types(self.u.atoms.types)
            self.u.add_TopologyAttr('elements', elements)

        ref_pro=mda.Universe(os.path.join(self.working_directory, self.pdb))
        if self.delete_units!='':
            reference= ref_pro.select_atoms('protein and (not (%s))' % self.delete_units)
        else:
            reference = ref_pro.select_atoms("protein")

        wat= self.u.select_atoms('resname HOH and resid 1')
        print('HOH 1:', len(wat))

        # --- 多条链完整处理: GroupHug 聚合 ---
        # 获取每条链
        prot_chain_list=[]
        for chain in self.u.select_atoms('protein').segments:
            if chain.atoms.n_atoms > 0:
                prot_chain_list.append(chain.atoms)
        #prot_chain_list = [chain.atoms for chain in self.u.segments if chain.atoms.n_atoms > 0]
        # GroupHug第一个参数为参考链，其余为需要聚合入同一个box的其它链
        print(prot_chain_list)
        prot_group = GroupHug(prot_chain_list[0], *prot_chain_list[1:])

        # ---- Transformation workflow ----
        self.workflow = (
            mda.transformations.unwrap(self.u.atoms,max_threads=10),
            prot_group,
            mda.transformations.center_in_box(self.u.select_atoms('protein'), center='geometry',max_threads=10),
            mda.transformations.wrap(self.u.select_atoms('protein or (%s)' % self.ligand_str), compound='fragments',max_threads=10), # LIG为小分子
            mda.transformations.wrap(self.u.select_atoms(self.solvent_str), compound='residues',max_threads=10),
            mda.transformations.fit_rot_trans(self.u.select_atoms('protein'), reference,max_threads=10),
        )
        self.u.trajectory.add_transformations(*self.workflow)

    def dihedral_angle(self,atom_label='',dist_step=10,ligand=True):
        if ligand:
            ligand_atom=self.u.select_atoms(self.ligand_str)
            for atom in ligand_atom:
                print(f"Atom ID: {atom.id}, Atom Name: {atom.name}, "
                      f"Residue Name: {atom.resname}{atom.resid}, "
                      f"Position: {atom.position}")
            if atom_label=='':
                atom_label=input('please input atom IDs for dihedral angle calculation (e.g. 2,3,4,5):\n')
            id1,id2,id3,id4=atom_label.split(',')
            ag_str='id %s or id %s or id %s or id %s' %(id1,id2,id3,id4)
        else:
            if atom_label=='':
                atom_label=input('please input atom IDs for dihedral angle calculation (e.g. resid 93 and (name CA or name CB or name CG or name C)):\n')
            ag_str=atom_label

        print(ag_str)
        ag=self.u.select_atoms(ag_str)
        from MDAnalysis.analysis.dihedrals import Dihedral
        R = Dihedral([ag]).run(start=None, stop=None, step=dist_step) 
        time_ids = [ts*self.dt for ts in self.u.trajectory]
        dihedrals=R.results.angles
        dihedrals=dihedrals.reshape(dihedrals.size)
        ary= np.vstack((np.array(time_ids[::dist_step]), dihedrals))
        df=pd.DataFrame(ary.T,columns=['time (ps)','dihedral angle'])
        df.to_csv(os.path.join(self.working_directory,'analysis','%s_dihedral.csv' % self.pdb))
        # Create a line plot using the values in dist and idx
        plt.hist(dihedrals, bins=60, density=True, alpha=0.6, color='g')

        # Add labels and title
        #plt.xlabel('Time (ps)')
        #plt.ylabel('Dihedral angle')
        plt.title('Dihedral angle distribution')

        # Save the figure to a file
        suffix=atom_label.replace(' ','_').replace(',','_')
        plt.savefig(os.path.join(self.working_directory,'analysis','%s_dihedral_analysis_%s.png' % (self.pdb,suffix)), dpi=300)
        plt.clf()
        plt.close('all')

    def calculate_angle_trajectory(self,atom_label=''):
        """
        计算三原子夹角随时间变化的函数
        参数:
            topology: 拓扑文件路径 (如.pdb, .gro等)
            trajectory: 轨迹文件路径 (如.xtc, .dcd等)
            atom_indices: 三个原子的索引列表 [atom1, atom2, atom3]
            output_file: 输出文件路径 (可选)
            plot_results: 是否绘制结果图
        返回:
            times: 时间数组 (ps)
            angles: 夹角数组 (度)
        """
        from MDAnalysis.lib.distances import calc_angles
        # 检查原子索引是否有效
        if atom_label=='':
            atom_label=input('please input atom label')
        atom_indices=[int(a) for a in atom_label.split(',')]
        for i, idx in enumerate(atom_indices):
            if idx < 0 or idx >= len(self.u.atoms):
                print(f"错误: 原子索引 {idx} 超出范围 (0-{len(u.atoms)-1})")
                return None, None

        atom1, atom2, atom3 = atom_indices
        print(f"计算原子 {atom1}-{atom2}-{atom3} 的夹角")
        print(f"原子名称: {self.u.atoms[atom1].name}-{self.u.atoms[atom2].name}-{self.u.atoms[atom3].name}")

        # 存储结果的列表
        times = []
        angles = []

        # 遍历轨迹计算夹角
        print("正在计算夹角...")
        for ts in self.u.trajectory:
            # 获取三个原子的坐标
            pos1 = self.u.atoms[atom1].position
            pos2 = self.u.atoms[atom2].position
            pos3 = self.u.atoms[atom3].position

            # 计算夹角 (返回弧度值)
            angle_rad = calc_angles(pos1, pos2, pos3, box=self.u.dimensions)

            # 转换为角度并存储
            angle_deg = np.degrees(angle_rad).flatten() # calc_angles返回数组
            times.append(ts.time)
            angles.append(angle_deg)

        times = np.array(times)
        angles = np.array(angles)

        # 输出统计信息
        print(f"\n夹角统计信息:")
        print(f"平均值: {np.mean(angles):.2f}°")
        print(f"标准差: {np.std(angles):.2f}°")
        print(f"最小值: {np.min(angles):.2f}°")
        print(f"最大值: {np.max(angles):.2f}°")

        """绘制夹角随时间变化的图形"""
        window_size = 10
        plt.figure(figsize=(10, 6))
        plt.plot(times, angles, 'b-', linewidth=1, alpha=0.7)
        #angle_smooth = np.convolve(angles, np.ones(window_size)/window_size, mode='valid')
        #plt.plot(time_ids[int(window_size/2):-int(window_size/2)+1], angle_smooth, label='Moving Average', color='red', linewidth=2)
        plt.xlabel('Time (ps)', fontsize=12)
        plt.ylabel('Angle (度)', fontsize=12)
        plt.title(f'Atom {atom_indices[0]}-{atom_indices[1]}-{atom_indices[2]} angle vs time', fontsize=14)
        plt.grid(True, alpha=0.3)

        # 添加统计信息文本框
        stats_text = f'平均值: {np.mean(angles):.2f}°\n标准差: {np.std(angles):.2f}°'
        plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction', 
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                     verticalalignment='top', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.working_directory,'analysis','%s_angle_analysis.png' % (self.pdb)), dpi=300)
        plt.clf()
        plt.close('all')
        #print(times,angles.ravel())
        ary= np.array((times, angles.ravel()))
        df=pd.DataFrame(ary.T,columns=['time (ps)','angle'])
        df.to_csv(os.path.join(self.working_directory,'analysis','%s_angle.csv' % self.pdb))

    def distance_analysis(self,atom_label='',dist_step=10):
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
            time_ids.append(idx*self.dt*dist_step)
            print(idx*self.dt*dist_step, dist_arr[0,0])

        ary=np.array([time_ids, dist])
        df=pd.DataFrame(ary.T,columns=['time (ns)','distance (A)'])

        # Create a line plot using the values in dist and idx
        plt.plot(time_ids, dist)

        window_size = 10
        distance_smooth = np.convolve(dist, np.ones(window_size)/window_size, mode='valid')
        plt.plot(time_ids[int(window_size/2):-int(window_size/2)+1], distance_smooth, label='Moving Average', color='red', linewidth=2)
        # Add labels and title
        plt.xlabel('Time (ps)')
        plt.ylabel('Distance ($\\AA$)')
        plt.title('Distance vs. Time (mean,std = %.2f,%.2f)' % (np.mean(np.array(dist)),np.std(np.array(dist))))

        # Save the figure to a file
        suffix=atom_label.replace(' ','_').replace(',','_')
        plt.savefig(os.path.join(self.working_directory,'analysis','%s_dist_analysis_%s.png' % (self.pdb,suffix)), dpi=300)
        plt.clf()
        plt.close('all')
        df.to_csv(os.path.join(self.working_directory,'analysis','%s_dist_analysis_%s.csv' % (self.pdb,suffix)))
        print('the result figure is saved in %s/analysis/%s_dist_analysis_%s.png\n' % (self.working_directory, self.pdb,suffix))
        return df

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

    def rmsd(self,rmsd_step=10):
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
        rmsd_df.index.name = 'Time (ns)'
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
        start_frame,end_frame=self.get_frames_by_time(start_time=start,end_time=end)
        #print(start_frame, end_frame + 1, slices)
        selected_frames= list(range(start_frame,end_frame))

        pro = universe.select_atoms('all')
        num_frames=len(selected_frames)

        if num_frames<slices:
            pro.write(os.path.join(self.working_directory,'analysis','%s_frames_%s.pdb' % (self.pdb,slices)), frames=selected_frames)
        else:
            pro.write(os.path.join(self.working_directory,'analysis','%s_frames_%s.pdb' % (self.pdb,slices)), frames=selected_frames[::int(num_frames/slices)])
        print('the result frames are saved in %s/analysis/%s_frames_(time).pdb\n' % (self.working_directory, self.pdb))

    def rmsf(self,selection='',ligand=True,align_str='protein and name CA',rmsf_step=10):
        """Calculate the RMSF of selected atomgroup in a MD simulation.

        Parameters
        ----------
        selection: str
        Selection string for the atomgroup to be investigated, also used during alignment.
        
        Output
        -------
        the plot of atom index and rmsf
        
        """
        if selection=='':
            lig_str=input('(1) ligand or (2) selection str for others\n')
            if lig_str == 'ligand' or lig_str == '1':
                selection=self.ligand_str
                ligand=True
                print('running ligand atom rmsf...')
            else:
                print('running protein residue rmsf...')
                selection = lig_str
                ligand=False

        pdbname=self.pdb.split('.')[0]

        protein_selection = self.u.select_atoms(selection)
        prealigner = align.AlignTraj(self.u, self.u, select=align_str,in_memory=True).run(step=rmsf_step)
        ref_coordinates = self.u.trajectory.timeseries(asel=protein_selection).mean(axis=1)
        self.ref_pro = mda.Merge(protein_selection).load_new(ref_coordinates[:, None, :],order="afc")
        aligner = align.AlignTraj(self.u, self.ref_pro, select=selection, in_memory=True).run(step=rmsf_step)

        ag = self.u.select_atoms(selection)
        R = rms.RMSF(ag).run(step=rmsf_step)
        if ligand:
            atoms_num=len(ag)
            
            with open(os.path.join(self.working_directory,'analysis','%s_ligand_rmsf.csv' % (pdbname)),'w') as f:
                for idx,lig_rmsf in zip(range(1,atoms_num+1), R.results.rmsf[0:atoms_num]):
                    f.write('%.3f,%.3f\n' % (idx, lig_rmsf))
            lig = range(1,atoms_num+1), R.results.rmsf[0:atoms_num]
            plt.plot(lig[0], lig[1],"ok-")
            plt.xlabel('Atom index')
            plt.ylabel('RMSF ($\AA$)')
            plt.savefig(os.path.join(self.working_directory,'analysis','%s_ligand_rmsf.png' % (pdbname)))
            plt.close('all')
        else:
            df = pd.DataFrame({"Residue ID": ag.resids, "Atom Name": ag.names, "rmsf": R.results.rmsf})
            df.to_csv(os.path.join(self.working_directory,'analysis','%s_rmsf.csv' % (pdbname)))
            plt.plot(ag.resids, R.results.rmsf)
            plt.xlabel('Residue number')
            plt.ylabel('RMSF of CA ($\AA$)')
            plt.legend()
            plt.savefig(os.path.join(self.working_directory,'analysis','%s_rmsf.png' % (pdbname)))
            
        print('the result figure is saved in %s/analysis/%s_rmsf.png\n' % (self.working_directory, self.pdb))

    def pro_lig_int(self,selection_str='',lig=True):
        """Calculate the protein-ligand interaction fingerprint using ProLif package.
        Notice
        -------
        1. If you use ProLIF in your research, please cite the following paper:
        Bouysset, C., Fiorucci, S. ProLIF: a library to encode molecular interactions as fingerprints.
        J Cheminform 13, 72 (2021). https://doi.org/10.1186/s13321-021-00548-6
        2. if you're loading a structure from a file that doesn't explicitely contain bond orders and formal charges, such as a PDB file or most MD trajectory files.
        MDAnalysis will infer those from the atoms connectivity, which requires all atoms including hydrogens to be present in the input file. In some cases,
        some atomic clashes may be incorrectly classified as bonds and will prevent the conversion of MDAnalysis molecules to RDKit.
        Since MDAnalysis uses van der Waals radii for bond detection, you can add the default radii if there is a atom is not recognized.
        ligand_selection.guess_bonds(vdwradii={"Cl": 1.75,"Br": 1.85,"I": 1.98})
        Returns
        -------
        1. plot of the protein-ligand interaction
        2. the fingerprint csv file that facilitate you to calculate the overall percent of precence of any interaction of interest.
        """
        if selection_str=='':
            ligand_selection=self.u.select_atoms(self.ligand_str)
        else:
            ligand_selection=self.u.select_atoms(selection_str)

        if lig:
            if '.pdb' in self.pdb or '.psf' in self.pdb:
                print('gussing elements...')
                from MDAnalysis.topology.guessers import guess_types
                elements = guess_types(self.u.atoms.names)
                self.u.add_TopologyAttr('elements', elements)
            print(ligand_selection)
            ligand_mol = plf.Molecule.from_mda(ligand_selection)
            pocket_selection = self.u.select_atoms("(protein) and byres around 6.0 group ligand",ligand=ligand_selection)
        else:
            if '.pdb' in self.pdb or '.psf' in self.pdb:
                ligand_selection.guess_bonds()
            pocket_selection = self.u.select_atoms("protein and not group peptide", peptide=ligand_selection)
            protein_selection = self.u.select_atoms("protein")
            if '.pdb' in self.pdb:
                pocket_selection.guess_bonds()

        fp = plf.Fingerprint()
        # run on a slice of the trajectory frames: from begining to end with a step of 10
        fp.run(self.u.trajectory[::100], ligand_selection, pocket_selection)
        if selection_str=='':
            sufix='lig'
        else:
            sufix=selection_str.replace(' ','_')
        #save the results
        fp.to_pickle(os.path.join(self.working_directory,'analysis','%s_fingerprint_%s.pkl' % (self.pdb.split('.')[0],sufix)))
        df = fp.to_dataframe()
        print(df)
        df.to_csv(os.path.join(self.working_directory,'analysis','%s_fingerprint_%s.csv' % (self.pdb.split('.')[0],sufix)))
        fp.plot_barcode()
        plt.savefig(os.path.join(self.working_directory,'analysis','%s_prolig_%s.png' % (self.pdb.split('.')[0],sufix)))
        plt.close('all')
        print('the result figure is saved in %s/analysis/%s_prolig_%s.png\n' % (self.working_directory,self.pdb.split('.')[0],sufix))

    def run(self):
        for file_name in os.listdir(self.working_directory):
            if "_system.pdb" in file_name or ".psf" in file_name or ".prmtop" in file_name:
                pdbfile=file_name
        for file_name in os.listdir(os.path.join(self.working_directory,'results')):
            if ".dcd" in file_name:
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
                'please select the mode:\n(1) rmsd\n(2) extract complex\n(3) distance analysis\n(4) ligand rmsf\n(5) protein-ligand fingerprint\n(6) dihedral angle analysis\n(7) protein-ligand heatmap\n(8) secondary structures\n(9) clustering analysis\n(0) quit\n')
            if job == '1':
                self.rmsd()
            elif job == '2':
                st, ed, num_frames = input(
                    'please input: starting time (ns),ending time(ns), number of frames\n').split(',')
                self.extract_complex(start=float(st), end=float(ed), slices=int(num_frames))
            elif job == '3':
                self.distance_analysis()
            elif job == '4':
                self.rmsf()
            elif job == '5':
                print('analyzing protein-ligand interaction...')
                lig_str=input('default ligand (Y) or usr defined ligand expression?\n')
                if lig_str=='Y':
                    lig_str=''
                    self.pro_lig_int(selection_str=lig_str)
                else:
                    lig_flag=input('if it is a ligand(1) or protein segment(2)?\n')
                    if lig_flag=='1':
                        self.pro_lig_int(selection_str=lig_str,lig=True)
                    else:
                        self.pro_lig_int(selection_str=lig_str,lig=False) 
            elif job == '6':
                print('analyzing dihedral angle...')
                lig_str=input('default ligand (Y) or residue (N)?\n')
                lig_flag=True
                if lig_str=='N':
                    lig_flag=False
                self.dihedral_angle(ligand=lig_flag)
            elif job == '7':
                print('analyzing protein-ligand heatmap...')
                self.heatmap()
            elif job == '8':
                print('analyzing secondary structures...')
                self.DSSP()
            elif job == '9':
                print('analyzing clustering analysis...')
                st, ed, n_cluster = input('please input:\nstarting time (ns),ending time(ns),number of clusters\n').split(',')
                self.cluster_md_trajectory(start=st, end=ed, n_clusters=n_cluster)
            elif job == '0':
                break

def test():
    work_dir='test'
    mda_ana=md_analysis(directory=work_dir)
    mda_ana.read_pdb_dcd('5o1c_14586_system.pdb', '5o1c_14586_2ns_new_trajectory.dcd',apo=False,remove_subunits='resname HOH',solv_selection='HOH',lig_selection='no')
    #mda_ana.calculate_angle_trajectory('20,21,22')
    #mda_ana.get_frames_by_time(start_time=0,end_time=100)
    #mda_ana.cluster_md_trajectory(start=0,end=10)
    mda_ana.distance_analysis()


if __name__ == '__main__':
    #test()
    work_dir=input('please input the name of working directory\n')
    mda_ana=md_analysis(directory=work_dir)
    mda_ana.run()
