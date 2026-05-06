# -*- coding: utf-8 -*-
"""
Flask web interface for MDAnalyzer.
Usage:
    conda run -n mda python web_app.py [--port 5050] [--host 0.0.0.0]
"""
import os
import sys
import json
import queue
import threading
import argparse
import traceback
from io import StringIO

from flask import Flask, render_template, request, jsonify, Response, send_from_directory

# Patch input() before importing md_analyzer so interactive prompts are no-ops
_input_queue = queue.Queue()
_output_log = []
_log_queue = queue.Queue()

def _patched_input(prompt=''):
    _log_queue.put({'type': 'prompt', 'text': str(prompt)})
    answer = _input_queue.get()
    _log_queue.put({'type': 'input_echo', 'text': str(answer)})
    return answer

import builtins
builtins.input = _patched_input

# Redirect print → log queue
_orig_stdout = sys.stdout

class _StreamCapture:
    def write(self, s):
        if s.strip():
            _log_queue.put({'type': 'log', 'text': s.rstrip()})
        _orig_stdout.write(s)
    def flush(self):
        _orig_stdout.flush()

sys.stdout = _StreamCapture()

# Now import analyzer
sys.path.insert(0, os.path.dirname(__file__))
from md_analyzer import MDAnalyzer

app = Flask(__name__, template_folder='templates')

# Global analyzer state
_analyzer: MDAnalyzer | None = None
_task_thread: threading.Thread | None = None
_task_running = False

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _run_in_thread(fn, answers: list[str]):
    """Run fn() in a thread, feeding *answers* to patched input() in order."""
    global _task_running
    _task_running = True
    # flush old items
    while not _input_queue.empty():
        _input_queue.get_nowait()

    for a in answers:
        _input_queue.put(a)

    def _worker():
        global _task_running
        try:
            fn()
            _log_queue.put({'type': 'done', 'text': '✔ Task complete.'})
        except Exception as e:
            _log_queue.put({'type': 'error', 'text': traceback.format_exc()})
        finally:
            _task_running = False

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def _working_dirs():
    base = os.path.join(os.path.dirname(__file__), 'working_directory')
    dirs = ['']  # root working_directory itself
    for entry in os.scandir(base):
        if entry.is_dir() and entry.name not in ('analysis', 'results'):
            dirs.append(entry.name)
    return dirs

# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', working_dirs=_working_dirs())


@app.route('/api/list_files', methods=['POST'])
def list_files():
    """Return topology and trajectory files for a given working directory."""
    directory = request.json.get('directory', '')
    base = os.path.join(os.path.dirname(__file__), 'working_directory')
    wd = os.path.join(base, directory) if directory else base
    results_dir = os.path.join(wd, 'results')

    topo_exts = ('_system.pdb', '.psf', '.prmtop', '.gro', '.pdb')
    topo = [f for f in os.listdir(wd) if any(f.endswith(e) for e in topo_exts)
            and os.path.isfile(os.path.join(wd, f))]
    traj = []
    if os.path.isdir(results_dir):
        traj = [f for f in os.listdir(results_dir)
                if any(f.endswith(e) for e in ('.dcd', '.xtc', '.trj', '.nc'))]
    return jsonify({'topo': sorted(topo), 'traj': sorted(traj)})


@app.route('/api/load', methods=['POST'])
def load():
    """Load topology + trajectory into the global analyzer."""
    global _analyzer
    data = request.json
    directory = data.get('directory', '')
    pdb = data.get('pdb', '')
    dcd = data.get('dcd', '')
    apo = data.get('apo', False)
    dt_override = data.get('dt', '')
    remove_subunits = data.get('remove_subunits', 'no')
    solv_selection = data.get('solv_selection', 'no')
    lig_selection = data.get('lig_selection', 'no')

    def _load():
        global _analyzer
        _analyzer = MDAnalyzer(directory=directory)
        _analyzer.read_pdb_dcd(
            pdb, dcd,
            apo=apo,
            remove_subunits=remove_subunits,
            solv_selection=solv_selection,
            lig_selection=lig_selection,
            dt=dt_override,
        )

    # dt answer fed to patched input if needed (read_pdb_dcd asks only when dt=='')
    answers = []
    if dt_override == '':
        answers.append('')  # keep default
    _run_in_thread(_load, answers)
    return jsonify({'status': 'started'})


@app.route('/api/universe_info', methods=['POST'])
def universe_info():
    if _analyzer is None:
        return jsonify({'error': 'No trajectory loaded'})
    _run_in_thread(_analyzer.universe_info, [])
    return jsonify({'status': 'started'})


@app.route('/api/rmsd', methods=['POST'])
def rmsd():
    if _analyzer is None:
        return jsonify({'error': 'No trajectory loaded'})
    data = request.json
    step = int(data.get('step', 10))
    _run_in_thread(lambda: _analyzer.rmsd(rmsd_step=step), [])
    return jsonify({'status': 'started'})


@app.route('/api/rmsf', methods=['POST'])
def rmsf():
    if _analyzer is None:
        return jsonify({'error': 'No trajectory loaded'})
    data = request.json
    mode = data.get('mode', 'ligand')   # 'ligand' | 'protein'
    selection = data.get('selection', '')
    align_str = data.get('align_str', 'protein and name CA')
    step = int(data.get('step', 10))

    if mode == 'ligand':
        answers = ['1']
    else:
        sel = selection if selection else 'protein and name CA'
        answers = ['2', sel]

    _run_in_thread(
        lambda: _analyzer.rmsf(align_str=align_str, rmsf_step=step),
        answers,
    )
    return jsonify({'status': 'started'})


@app.route('/api/distance', methods=['POST'])
def distance():
    if _analyzer is None:
        return jsonify({'error': 'No trajectory loaded'})
    data = request.json
    atom_label = data.get('atom_label', '')
    step = int(data.get('step', 10))
    if not atom_label:
        return jsonify({'error': 'atom_label required (two comma-separated selections)'})
    _run_in_thread(
        lambda: _analyzer.distance_analysis(atom_label=atom_label, dist_step=step),
        [],
    )
    return jsonify({'status': 'started'})


@app.route('/api/dihedral', methods=['POST'])
def dihedral():
    if _analyzer is None:
        return jsonify({'error': 'No trajectory loaded'})
    data = request.json
    atom_label = data.get('atom_label', '')
    step = int(data.get('step', 10))
    ligand = data.get('ligand', True)
    if not atom_label:
        return jsonify({'error': 'atom_label required'})
    _run_in_thread(
        lambda: _analyzer.dihedral_angle(
            atom_label=atom_label, dist_step=step, ligand=ligand),
        [],
    )
    return jsonify({'status': 'started'})


@app.route('/api/angle', methods=['POST'])
def angle():
    if _analyzer is None:
        return jsonify({'error': 'No trajectory loaded'})
    data = request.json
    atom_label = data.get('atom_label', '')
    if not atom_label:
        return jsonify({'error': 'atom_label required (three comma-separated atom indices)'})
    _run_in_thread(
        lambda: _analyzer.calculate_angle_trajectory(atom_label=atom_label),
        [],
    )
    return jsonify({'status': 'started'})


@app.route('/api/extract', methods=['POST'])
def extract():
    if _analyzer is None:
        return jsonify({'error': 'No trajectory loaded'})
    data = request.json
    start = float(data.get('start', 0))
    end = float(data.get('end', 0))
    slices = int(data.get('slices', 50))
    _run_in_thread(
        lambda: _analyzer.extract_complex(start=start, end=end, slices=slices),
        [],
    )
    return jsonify({'status': 'started'})


@app.route('/api/dssp', methods=['POST'])
def dssp():
    if _analyzer is None:
        return jsonify({'error': 'No trajectory loaded'})
    data = request.json
    residues = data.get('residues', 'all')
    step = int(data.get('step', 50))
    _run_in_thread(
        lambda: _analyzer.DSSP(residues=residues, dssp_step=step),
        [],
    )
    return jsonify({'status': 'started'})


@app.route('/api/heatmap', methods=['POST'])
def heatmap():
    if _analyzer is None:
        return jsonify({'error': 'No trajectory loaded'})
    _run_in_thread(_analyzer.heatmap, [])
    return jsonify({'status': 'started'})


@app.route('/api/fingerprint', methods=['POST'])
def fingerprint():
    if _analyzer is None:
        return jsonify({'error': 'No trajectory loaded'})
    data = request.json
    selection_str = data.get('selection_str', '')
    lig = data.get('lig', True)
    _run_in_thread(
        lambda: _analyzer.pro_lig_int(selection_str=selection_str, lig=lig),
        [],
    )
    return jsonify({'status': 'started'})


@app.route('/api/cluster', methods=['POST'])
def cluster():
    if _analyzer is None:
        return jsonify({'error': 'No trajectory loaded'})
    data = request.json
    start = float(data.get('start', 0))
    end = float(data.get('end', 0))
    n_clusters = int(data.get('n_clusters', 5))
    selection_mode = data.get('selection_mode', '2')  # '1','2','3'
    custom_sel = data.get('custom_selection', '')

    answers = [selection_mode]
    if selection_mode == '3':
        answers.append(custom_sel)

    _run_in_thread(
        lambda: _analyzer.cluster_md_trajectory(
            start=start, end=end, n_clusters=n_clusters),
        answers,
    )
    return jsonify({'status': 'started'})


@app.route('/api/stream')
def stream():
    """SSE endpoint: push log lines to the browser."""
    def _generate():
        yield 'data: {"type":"connected","text":"Log stream connected."}\n\n'
        while True:
            try:
                msg = _log_queue.get(timeout=30)
                yield f'data: {json.dumps(msg)}\n\n'
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'
    return Response(_generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/status')
def status():
    loaded = _analyzer is not None
    info = {}
    if loaded:
        try:
            info = {
                'pdb': _analyzer.pdb,
                'dcd': _analyzer.dcd,
                'apo': _analyzer.apo,
                'dt': _analyzer.dt,
                'n_frames': len(_analyzer.u.trajectory),
                'n_atoms': _analyzer.u.atoms.n_atoms,
                'ligand_str': _analyzer.ligand_str,
            }
        except Exception:
            pass
    return jsonify({'loaded': loaded, 'running': _task_running, 'info': info})


@app.route('/api/output_files')
def output_files():
    """List all files in analysis/ directories."""
    files = []
    base = os.path.join(os.path.dirname(__file__), 'working_directory')
    for root, dirs, fnames in os.walk(base):
        if 'analysis' in root.split(os.sep):
            for f in fnames:
                rel = os.path.relpath(os.path.join(root, f), base)
                files.append(rel)
    return jsonify({'files': sorted(files)})


@app.route('/api/view_file')
def view_file():
    rel = request.args.get('path', '')
    base = os.path.join(os.path.dirname(__file__), 'working_directory')
    full = os.path.normpath(os.path.join(base, rel))
    if not full.startswith(base):
        return jsonify({'error': 'forbidden'}), 403
    directory = os.path.dirname(full)
    filename = os.path.basename(full)
    return send_from_directory(directory, filename)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MD Analyzer Web UI')
    parser.add_argument('--port', type=int, default=5050)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print(f'Starting MD Analyzer Web UI at http://{args.host}:{args.port}')
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
