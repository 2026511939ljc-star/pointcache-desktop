"""Windows desktop front end for the bundled point cache engine."""

from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


WORKER = Path(__file__).resolve().parent / 'engine' / 'worker.py'
BG = '#101820'
PANEL = '#17232e'
PLOT = '#121e29'
INK = '#edf4fb'
MUTED = '#a9bfd0'
CYAN = '#50dec9'
GRID = '#2b4051'
FONT = ('Microsoft YaHei UI', 10)
MODES = {'隔帧抽取': 'stride', '隔帧＋关键采样（HBJSON）': 'key_samples', '逐 ID 自适应': 'adaptive'}


def find_hython():
    folders = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Side Effects Software'
    options = {}
    for path in folders.glob('Houdini */bin/hython.exe'):
        version = path.parents[1].name.split()[-1].split('.')
        if path.is_file() and all(part.isdecimal() for part in version):
            options[tuple(map(int, version))] = path
    return str(options[max(options)]) if options else ''


def cache_kind(path):
    return 'hbjson' if path.is_file() and path.suffix.lower() == '.hbjson' else 'bgeo'


def run_job(source, output, options, hython):
    python = Path(sys.executable)
    if python.name.lower() == 'pythonw.exe':
        python = python.with_name('python.exe')
    if cache_kind(source) == 'bgeo':
        if not hython or not Path(hython).is_file():
            raise ValueError('请选择 Houdini 安装目录 bin 下的 hython.exe。')
        python = Path(hython)
    config = {**options, 'source': str(source), 'output': str(output)}
    if cache_kind(source) == 'hbjson':
        config['format'] = 'hbjson'
    with tempfile.TemporaryDirectory(prefix='pointcache-job-') as temp:
        job_file = Path(temp) / 'job.json'
        job_file.write_text(json.dumps(config, ensure_ascii=False), encoding='utf-8')
        result = subprocess.run(
            [str(python), '-X', 'utf8', str(WORKER), str(job_file)],
            capture_output=True, encoding='utf-8', errors='replace',
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
    if result.returncode:
        error = result.stderr.strip() or result.stdout.strip() or f'退出码 {result.returncode}'
        if 'No module named' in error and 'numpy' in error:
            error = '缓存处理所用 Python 缺少 NumPy。\n' + error
        raise RuntimeError(error)
    return json.loads((output / 'validation.json').read_text(encoding='utf-8'))


class App:
    def __init__(self, root):
        self.root = root
        self.root.title('点云缓存工作台')
        self.root.geometry('1180x790')
        self.root.minsize(1000, 765)
        self.root.configure(bg=BG)
        self.root.option_add('*Font', FONT)
        self.sources = {}
        self.results = {}
        self.messages = queue.Queue()
        self.running = False
        self.chart_report = None
        self.chart_index = None
        self.open_directory = None
        self.open_validation = None
        self.method = tk.StringVar(value='隔帧抽取')
        self.output = tk.StringVar()
        self.hython = tk.StringVar(value=find_hython())
        self.format = tk.StringVar(value='bgeo')
        self.stride = tk.StringVar(value='2')
        self.fps = tk.StringVar()
        self.pattern = tk.StringVar(value='*.bgeo.sc')
        self.boundaries = tk.BooleanVar()
        self.pack = tk.BooleanVar()
        self.position = tk.StringVar(value='0.03')
        self.velocity = tk.StringVar(value='0.5')
        self.turns = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value='等待添加缓存')
        self.cursor_text = tk.StringVar()
        self.metrics = [tk.StringVar(value='—') for _ in range(4)]
        self.style()
        self.build()
        self.toggle_options()
        self.root.protocol('WM_DELETE_WINDOW', self.close)

    def style(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', font=FONT, background=PANEL, foreground=INK,
                        fieldbackground=PLOT, bordercolor=GRID, lightcolor=GRID, darkcolor=GRID)
        style.configure('TFrame', background=PANEL)
        style.configure('Outer.TFrame', background=BG)
        style.configure('Muted.TLabel', foreground=MUTED)
        style.configure('TEntry', insertcolor=INK, padding=5)
        style.configure('TCombobox', padding=4, arrowcolor=CYAN)
        style.map('TCombobox', fieldbackground=[('readonly', PLOT)], foreground=[('readonly', INK)])
        style.configure('TButton', background='#283a4a', padding=(9, 5))
        style.map('TButton', background=[('active', '#385169')])
        style.configure('Accent.TButton', background=CYAN, foreground=BG, padding=(12, 6))
        style.map('Accent.TButton', background=[('active', '#8df2e3'), ('disabled', '#344657')])
        style.configure('Treeview', background=PLOT, fieldbackground=PLOT, rowheight=27)
        style.map('Treeview', background=[('selected', '#285768')], foreground=[('selected', INK)])
        style.configure('Treeview.Heading', background='#233545', foreground=MUTED)
        style.configure('Horizontal.TProgressbar', background=CYAN, troughcolor=PLOT)

    def build(self):
        top = tk.Frame(self.root, bg=BG, padx=24, pady=9)
        top.pack(fill='x')
        tk.Label(top, text='点云缓存工作台', font=(FONT[0], 23, 'bold'),
                 fg=INK, bg=BG).pack(side='left')
        tk.Label(top, text='Houdini / Niagara', font=('Consolas', 13),
                 fg=CYAN, bg=BG).pack(side='right')
        bottom = ttk.Frame(self.root, style='Outer.TFrame', padding=(22, 0, 22, 10))
        bottom.pack(side='bottom', fill='x')
        ttk.Label(bottom, textvariable=self.status, background=BG, foreground=MUTED).pack(side='left')
        self.progress = ttk.Progressbar(bottom, length=180, mode='determinate')
        self.progress.pack(side='right')
        body = ttk.Frame(self.root, style='Outer.TFrame', padding=(22, 0, 22, 8))
        body.pack(expand=True, fill='both')
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body, padding=13)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 12))
        left.columnconfigure(1, weight=1)
        right = ttk.Frame(body, padding=13)
        right.grid(row=0, column=1, sticky='nsew')
        self.build_controls(left)
        self.build_results(right)

    def path_field(self, parent, row, label, value, action):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=3)
        ttk.Entry(parent, textvariable=value, width=23).grid(row=row, column=1, sticky='ew', padx=6)
        ttk.Button(parent, text='选择', command=action).grid(row=row, column=2)

    def build_controls(self, frame):
        ttk.Label(frame, text='缓存队列', font=(FONT[0], 13, 'bold')).grid(row=0, columnspan=3, sticky='w')
        self.tree = ttk.Treeview(frame, columns=('state',), height=3, selectmode='browse')
        self.tree.heading('#0', text='输入')
        self.tree.heading('state', text='状态')
        self.tree.column('#0', width=228, stretch=True)
        self.tree.column('state', width=75, stretch=False)
        self.tree.grid(row=1, columnspan=2, sticky='ew', pady=5)
        scroll = ttk.Scrollbar(frame, orient='vertical', command=self.tree.yview)
        scroll.grid(row=1, column=2, sticky='nsw', pady=5)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.bind('<<TreeviewSelect>>', self.select_result)
        buttons = ttk.Frame(frame)
        buttons.grid(row=2, columnspan=3, sticky='w')
        self.queue_controls = []
        for title, action in [('添加文件', self.add_files), ('添加序列目录', self.add_folder),
                              ('移除选中', self.remove)]:
            button = ttk.Button(buttons, text=title, command=action)
            button.pack(side='left', padx=(0, 4))
            self.queue_controls.append(button)

        ttk.Separator(frame).grid(row=3, columnspan=3, sticky='ew', pady=7)
        self.path_field(frame, 4, '输出目录', self.output, self.choose_output)
        self.path_field(frame, 5, 'hython', self.hython, self.choose_hython)
        ttk.Label(frame, text='处理方式').grid(row=6, column=0, sticky='w', pady=6)
        method = ttk.Combobox(frame, textvariable=self.method, values=list(MODES), state='readonly')
        method.grid(row=6, column=1, columnspan=2, sticky='ew', padx=(6, 0))
        method.bind('<<ComboboxSelected>>', lambda _: self.toggle_options())
        ttk.Label(frame, text='BGEO 导出').grid(row=7, column=0, sticky='w')
        ttk.Combobox(frame, textvariable=self.format, values=['bgeo', 'hbjson', 'both'],
                     state='readonly', width=12).grid(row=7, column=1, sticky='w', padx=6)

        options = ttk.Frame(frame)
        options.grid(row=8, columnspan=3, sticky='ew', pady=(8, 1))
        for row, (title, var) in enumerate([('每 N 帧取一帧', self.stride),
                                            ('位置误差（源单位）', self.position), ('速度误差（源单位）', self.velocity),
                                            ('缺少时间时的 FPS', self.fps), ('BGEO 文件匹配', self.pattern)]):
            ttk.Label(options, text=title).grid(row=row, column=0, sticky='w')
            entry = ttk.Entry(options, textvariable=var, width=15)
            entry.grid(row=row, column=1, sticky='w', padx=8)
            if row == 0:
                self.stride_input = entry
            elif row == 1:
                self.position_input = entry
            elif row == 2:
                self.velocity_input = entry
        self.boundary_input = ttk.Checkbutton(frame, text='保留每个 ID 首末记录所在整帧',
                                              variable=self.boundaries)
        self.boundary_input.grid(row=9, columnspan=3, sticky='w', pady=(3, 0))
        self.turn_input = ttk.Checkbutton(frame, text='补回位置转折采样', variable=self.turns)
        self.turn_input.grid(row=10, columnspan=3, sticky='w')
        ttk.Checkbutton(frame, text='同时生成传输 ZIP', variable=self.pack).grid(row=11, columnspan=3, sticky='w')
        actions = ttk.Frame(frame)
        actions.grid(row=12, columnspan=3, sticky='w', pady=(10, 0))
        self.inspect_button = ttk.Button(actions, text='只分析', command=lambda: self.start(True))
        self.inspect_button.pack(side='left', padx=(0, 6))
        self.run_button = ttk.Button(actions, text='生成测试缓存', style='Accent.TButton',
                                     command=lambda: self.start(False))
        self.run_button.pack(side='left')

    def build_results(self, frame):
        ttk.Label(frame, text='结果对照', font=(FONT[0], 13, 'bold')).pack(anchor='w')
        numbers = tk.Frame(frame, bg=PANEL, pady=10)
        numbers.pack(fill='x')
        for i, title in enumerate(['缓存体积', '采样记录', '粒子 ID', '原始时间范围']):
            numbers.columnconfigure(i, weight=1, uniform='numbers')
            card = tk.Frame(numbers, bg=PLOT, padx=9, pady=7)
            card.grid(row=0, column=i, padx=(0, 5), sticky='nsew')
            tk.Label(card, text=title, bg=PLOT, fg=MUTED, font=(FONT[0], 9)).pack(anchor='w')
            tk.Label(card, textvariable=self.metrics[i], bg=PLOT, fg=INK,
                     font=(FONT[0], 10, 'bold')).pack(anchor='w', pady=(4, 0))
        self.chart = tk.Canvas(frame, background=PLOT, height=225, highlightthickness=0)
        self.chart.pack(fill='x')
        self.chart.bind('<Configure>', lambda _: self.draw_chart())
        self.chart.bind('<Motion>', self.inspect_chart)
        ttk.Label(frame, textvariable=self.cursor_text, style='Muted.TLabel').pack(anchor='w', pady=4)
        self.details = tk.Text(frame, bg=PLOT, fg=INK, relief='flat', height=8, wrap='word',
                               font=FONT, padx=10, pady=8, state='disabled')
        self.details.pack(fill='both', expand=True, pady=(5, 7))
        actions = ttk.Frame(frame)
        actions.pack(fill='x')
        self.folder_button = ttk.Button(actions, text='打开结果目录',
                                        command=lambda: self.open_result(self.open_directory), state='disabled')
        self.report_button = ttk.Button(actions, text='打开验证报告',
                                        command=lambda: self.open_result(self.open_validation), state='disabled')
        self.folder_button.pack(side='left', padx=(0, 6))
        self.report_button.pack(side='left')
        self.set_details('添加缓存后，点击「只分析」读取帧数、粒子 ID 和时间范围。')

    def toggle_options(self):
        mode = MODES[self.method.get()]
        self.stride_input.configure(state='normal' if mode == 'stride' else 'disabled')
        for widget in (self.position_input, self.velocity_input):
            widget.configure(state='normal' if mode != 'stride' else 'disabled')
        self.turn_input.configure(state='normal' if mode == 'key_samples' else 'disabled')
        self.boundary_input.configure(state='normal' if mode == 'stride' else 'disabled')

    def add_files(self):
        for name in filedialog.askopenfilenames(filetypes=[('缓存或序列 ZIP', '*.hbjson *.zip'),
                                                            ('所有文件', '*.*')]):
            self.add_source(Path(name))

    def add_folder(self):
        name = filedialog.askdirectory()
        if name:
            self.add_source(Path(name))

    def add_source(self, path):
        path = path.resolve()
        if path.is_file() and path.suffix.lower() not in ('.zip', '.hbjson'):
            messagebox.showerror('文件格式', '请选择 HBJSON、BGEO 序列 ZIP，或添加整个 BGEO 序列目录。')
            return
        if path not in self.sources.values():
            item = self.tree.insert('', 'end', text=path.name, values=('待处理',))
            self.sources[item] = path
            if not self.output.get():
                self.output.set(str(path.parent / 'PointCacheResults'))
            self.tree.selection_set(item)

    def choose_output(self):
        name = filedialog.askdirectory()
        if name:
            self.output.set(name)

    def choose_hython(self):
        name = filedialog.askopenfilename(filetypes=[('Houdini Python', 'hython.exe'), ('程序', '*.exe')])
        if name:
            self.hython.set(name)

    def remove(self):
        for item in self.tree.selection():
            self.tree.delete(item)
            self.sources.pop(item, None)
            self.results.pop(item, None)
        self.select_result()

    def options(self, inspect):
        mode = 'inspect' if inspect else MODES[self.method.get()]
        options = dict(mode=mode, format=self.format.get(), pack=self.pack.get() and not inspect,
                       pattern=self.pattern.get().strip() or '*.bgeo.sc')
        if self.fps.get().strip() and any(cache_kind(p) == 'bgeo' for p in self.sources.values()):
            options['fps'] = float(self.fps.get())
            if not math.isfinite(options['fps']) or not options['fps'] > 0:
                raise ValueError('FPS 必须大于零。')
        if mode == 'stride':
            options.update(stride=int(self.stride.get()), preserve_boundaries=self.boundaries.get())
            if options['stride'] < 2:
                raise ValueError('抽帧间隔 N 至少为 2。')
        if mode in ('adaptive', 'key_samples'):
            options.update(position_tolerance=float(self.position.get()),
                           velocity_tolerance=float(self.velocity.get()), velocity_relative=0.0,
                           protect_turns=self.turns.get())
            if any(not math.isfinite(options[key]) or options[key] < 0
                   for key in ('position_tolerance', 'velocity_tolerance')):
                raise ValueError('误差上限需要是非负有限数值。')
        if mode == 'key_samples' and any(cache_kind(p) == 'bgeo' for p in self.sources.values()):
            raise ValueError('关键采样方式适用于 HBJSON。BGEO 可以先用隔帧或自适应方式处理。')
        return options

    def start(self, inspect=False):
        if self.running:
            return
        if not self.sources or not self.output.get().strip():
            messagebox.showinfo('开始处理', '请先添加缓存并选择输出目录。')
            return
        try:
            options = self.options(inspect)
            sources = list(self.sources.items())
            output_root = Path(self.output.get().strip()).expanduser().resolve()
            for _, source in sources:
                if source == output_root or source in output_root.parents:
                    raise ValueError('输出目录需要放在输入序列目录之外。')
            destination = output_root / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        except (ValueError, OSError) as error:
            messagebox.showerror('检查设置', str(error))
            return
        self.running = True
        self.progress.configure(value=0, maximum=len(sources))
        self.status.set('正在读取缓存…')
        for control in (self.run_button, self.inspect_button, *self.queue_controls):
            control.configure(state='disabled')
        for item, _ in sources:
            self.tree.set(item, 'state', '等待')
            self.results.pop(item, None)
        if not self.tree.selection():
            self.tree.selection_set(sources[0][0])
        self.select_result()
        threading.Thread(target=self.process_queue, args=(sources, destination, options, self.hython.get()),
                         daemon=True).start()
        self.root.after(100, self.poll)

    def process_queue(self, sources, destination, options, hython):
        completed = 0
        for index, (item, source) in enumerate(sources, 1):
            self.messages.put(('start', item, source.name))
            output = destination / f'{index:03d}_{source.stem[:65]}'
            try:
                report = run_job(source, output, options, hython)
            except Exception as error:
                self.messages.put(('error', item, str(error)))
            else:
                completed += 1
                self.messages.put(('done', item, (output, report)))
        self.messages.put(('finish', completed, len(sources)))

    def poll(self):
        try:
            while True:
                event, item, data = self.messages.get_nowait()
                if event == 'start':
                    self.tree.set(item, 'state', '处理中')
                    self.status.set(f'正在处理 {data}')
                elif event in ('done', 'error'):
                    self.results[item] = data
                    self.tree.set(item, 'state', '完成' if event == 'done' else '失败')
                    self.progress.configure(value=float(self.progress['value']) + 1)
                    if self.tree.selection() == (item,):
                        self.select_result()
                else:
                    self.running = False
                    for control in (self.run_button, self.inspect_button, *self.queue_controls):
                        control.configure(state='normal')
                    self.status.set(f'完成 {item} 项 · 失败 {data - item} 项')
                    return
        except queue.Empty:
            self.root.after(100, self.poll)

    def select_result(self, _event=None):
        selected = self.tree.selection()
        result = self.results.get(selected[0]) if selected else None
        self.chart_report = None
        self.chart_index = None
        self.cursor_text.set('')
        for metric in self.metrics:
            metric.set('—')
        for control in (self.folder_button, self.report_button):
            control.configure(state='disabled')
        if not isinstance(result, tuple):
            source = self.sources.get(selected[0]) if selected else None
            self.set_details('\n\n'.join(str(x) for x in (source, result) if x)
                             or '选择队列中的缓存查看结果。')
            self.draw_chart()
            return
        output, report = result
        self.open_directory = output
        self.open_validation = output / 'validation.json'
        self.folder_button.configure(state='normal')
        self.report_button.configure(state='normal')
        self.chart_report = report
        self.show_metrics(report, output)
        self.draw_chart()

    def show_metrics(self, report, output):
        life = report['lifecycle']
        analysis = report['analysis']
        inspecting = report['mode'] == 'inspect'
        source, retained = report['source_samples'], report['output_samples']
        start, end = report['original_time_range']
        self.metrics[1].set(f'{source:,}' if inspecting else f'{source:,}\n→ {retained:,}')
        self.metrics[2].set(f'{life["retained_IDs"]:,}')
        self.metrics[3].set(f'{start:.3f} – {end:.3f} s')
        source_total = sum(f['bytes'] for f in report['source_files'])
        target_total = sum(f['bytes'] for f in report['outputs'])
        fmt = report['config']['format']
        comparable = (report['source_kind'], fmt) in (('hbjson', 'hbjson'), ('bgeo', 'bgeo'))
        self.metrics[0].set(f'{source_total / 1048576:.1f} MiB' if inspecting else
                            f'{source_total / 1048576:.1f} → {target_total / 1048576:.1f}\nMiB')
        text = [f'帧数 {report["source_frames"]} → {report["output_frames"]}  ·  丢失 ID {life["lost_IDs"]:,}']
        if not inspecting:
            text.append(f'采样减少 {100 * (1 - retained / source):.1f}%')
            if comparable and source_total:
                delta = 100 * (1 - target_total / source_total)
                direction = '减少' if delta >= 0 else '增加'
                text[-1] += f'  ·  同格式体积{direction} {abs(delta):.1f}%'
            if not comparable:
                text.append(f'体积对比使用源数据与 {fmt.upper()} 输出。')
            if report.get('source_container'):
                text.append('源体积按 ZIP 内解包后的 BGEO 序列计算。')
        text.append(f'首样本偏移 {life["first_available_sample_changed_IDs"]:,} 个 ID  ·  '
                    f'末样本偏移 {life["last_available_sample_changed_IDs"]:,} 个 ID')
        if 'supplemented_records' in analysis:
            text.append(f'补回关键记录 {analysis["supplemented_records"]:,} 条')
        if 'samples_without_both_ID_neighbors' in analysis:
            text.append(f'缺少同 ID 前后邻点的记录 {analysis["samples_without_both_ID_neighbors"]:,} 条')
        for name in ('P', 'v'):
            error = analysis.get('attribute_errors', {}).get(name)
            if error and error['samples']:
                text.append(f'{name} 插值误差  ·  P95 {error["p95"]:.5g}  ·  最大 {error["max"]:.5g}（源单位）')
        if (output / 'result.zip').is_file():
            text.append(f'传输 ZIP {(output / "result.zip").stat().st_size / 1048576:.1f} MiB')
        if not inspecting:
            text.append('数据验证通过。用实际播放效果验收。')
        self.set_details('\n'.join(text))

    def set_details(self, value):
        self.details.configure(state='normal')
        self.details.delete('1.0', 'end')
        self.details.insert('1.0', value)
        self.details.configure(state='disabled')

    def inspect_chart(self, event):
        if not self.chart_report:
            return
        frames = self.chart_report['source_timeline']
        start, end = frames[0]['time'], frames[-1]['time']
        ratio = (event.x - 52) / max(1, self.chart.winfo_width() - 70)
        time = start + ratio * (end - start)
        self.chart_index = min(range(len(frames)), key=lambda i: abs(frames[i]['time'] - time))
        self.draw_chart()

    def draw_chart(self):
        chart = self.chart
        chart.delete('all')
        width, height = chart.winfo_width(), chart.winfo_height()
        if not self.chart_report:
            chart.create_text(width / 2, height / 2, text='处理完成后显示各时间点的采样记录',
                              fill=MUTED, font=FONT)
            return
        report = self.chart_report
        frames = report['source_timeline']
        start, end = frames[0]['time'], frames[-1]['time']
        peak = max(1, max(frame['source_points'] for frame in frames))
        left, right, top, base = 52, max(53, width - 18), 40, height - 26
        def x(t):
            return left + (t - start) / (end - start or 1) * (right - left)
        def y(count):
            return base - count / peak * (base - top)
        chart.create_text(12, 15, text='每个时间点的采样记录', fill=INK, anchor='w', font=FONT)
        chart.create_text(right, 15, text='灰线  原始    绿点  保留', fill=MUTED, anchor='e', font=FONT)
        for j in range(5):
            count = peak * j / 4
            chart.create_line(left, y(count), right, y(count), fill=GRID)
            label = str(round(count)) if peak < 1000 else f'{count / 1000:.0f}k'
            chart.create_text(left - 6, y(count), text=label, fill=MUTED, anchor='e')
        for t in (start, end) if end != start else (start,):
            chart.create_text(x(t), height - 11, text=f'{t:.3f}s', fill=MUTED)
        coords = [coord for frame in frames for coord in (x(frame['time']), y(frame['source_points']))]
        if len(coords) > 2:
            chart.create_line(*coords, fill=MUTED, width=2)
        inspecting = report['mode'] == 'inspect'
        retained = {frame['number']: frame['retained_samples'] for frame in report['frames']}
        if not inspecting or len(coords) == 2:
            for frame in frames:
                count = retained.get(frame['number'])
                if count is not None:
                    cx, cy = x(frame['time']), y(count)
                    chart.create_oval(cx - 2, cy - 2, cx + 2, cy + 2, fill=CYAN, outline='')
        if self.chart_index is not None:
            selected = frames[self.chart_index]
            cx = x(selected['time'])
            chart.create_line(cx, top, cx, base, fill=CYAN, dash=(3, 3))
            count = retained.get(selected['number'], 0)
            self.cursor_text.set(f'第 {selected["number"]} 帧  ·  {selected["time"]:.4f} s  ·  '
                                 f'原始 {selected["source_points"]:,} 条  ·  保留 {count:,} 条')

    @staticmethod
    def open_result(path):
        try:
            os.startfile(path)
        except OSError as error:
            messagebox.showerror('打开失败', str(error))

    def close(self):
        if self.running:
            self.status.set('当前队列仍在处理，完成后可关闭。')
        else:
            self.root.destroy()


if __name__ == '__main__':
    window = tk.Tk()
    App(window)
    window.mainloop()
