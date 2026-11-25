"""Batch BGEO.SC/HBJSON temporal reduction. Python 3.10+, NumPy; BGEO needs hython.

No Unreal API, overwrite, ID renumbering, coordinate rescaling or time rescaling.
See --help and references/batch-usage.md. Kept HBJSON records are byte-identical.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import fnmatch
import glob
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import sys
import tempfile
import zipfile
import numpy as np
import pointcache as pc

VERSION = '2.0.0'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def get_hou():
    try:
        import hou
        return hou
    except ImportError as e:
        raise ValueError('BGEO requires Houdini hython.exe with NumPy; run this script using hython.') from e


def text(s):
    b = s.encode('utf-8')
    return b'B'+struct.pack('<B', len(b))+b if len(b)<256 else b'L'+struct.pack('<I', len(b))+b


def header_bytes(attrs, counts):
    b = b'{'+text('header')+b'{'+text('version')+text('1.0')
    for name in ('num_samples', 'num_frames', 'num_points'):
        require(0 <= counts[name] < 2**32, 'HBJSON count exceeds uint32.')
        b += text(name)+struct.pack('<I', counts[name])
    require(len(attrs)<2**16, 'Too many attributes.')
    b += text('num_attrib')+struct.pack('<H', len(attrs))
    b += text('attrib_name')+b'['+b''.join(text(a['name']) for a in attrs)+b']'
    b += text('attrib_size')+b'['+bytes(a['size'] for a in attrs)+b']'
    b += text('attrib_data_type')+b'['+''.join(a['types'] for a in attrs).encode('ascii')+b']'
    return b+text('data_type')+text('linear')+b'}'+text('cache_data')+b'{'+text('frames')+b'['


@dataclass
class Frame:
    number: int
    time: float
    rows: np.ndarray
    path: Path | None = None


@dataclass
class Cache:
    frames: list
    attrs: list
    dtype: np.dtype
    sources: list
    kind: str


def source_stamp(path):
    p = Path(path)
    return dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=digest(p))


def safe_extract(source, folder, pattern, max_bytes):
    with zipfile.ZipFile(source) as z:
        members = z.infolist()
        require(sum(m.file_size for m in members) <= max_bytes, 'ZIP extraction exceeds --max-extract-gib.')
        selected = []
        seen = set()
        for m in members:
            p = PurePosixPath(m.filename.replace('\\', '/'))
            require(not p.is_absolute() and '..' not in p.parts and ':' not in str(p), 'Unsafe ZIP path.')
            require((m.external_attr >> 16) & 0o170000 != 0o120000, 'ZIP symlinks are unsupported.')
            if m.is_dir() or not fnmatch.fnmatch(p.name, pattern):
                continue
            require(p.name.casefold() not in seen, 'Duplicate sequence basename inside ZIP.')
            seen.add(p.name.casefold())
            target = folder/p.name
            with z.open(m) as inp, target.open('xb') as out:
                shutil.copyfileobj(inp, out)
            selected.append(target)
        return selected


def sequence_order(paths):
    result = []
    groups = set()
    for p in paths:
        match = re.fullmatch(r'(.*?)(-?\d+)(\.bgeo(?:\.sc)?)', p.name, re.I)
        require(match is not None, 'BGEO filename needs a numeric frame suffix: '+p.name)
        groups.add((match[1], match[3].lower()))
        result.append((int(match[2]), p))
    require(len(groups)==1 and result, 'Select exactly one BGEO sequence per job; narrow --pattern or split batch jobs.')
    result.sort(key=lambda pair: pair[0])
    require(len({n for n, _ in result})==len(result), 'Duplicate frame labels.')
    return result


def bgeo_cache(paths, config):
    hou = get_hou()
    frames, attrs, dtype, signature = [], None, None, None
    stamps = []
    retained_bytes = 0
    code_map = dict(Int8='b', Int16='h', Int32='l', Int64='q', Float32='f', Float64='d')
    for number, path in sequence_order(paths):
        g = hou.Geometry()
        g.loadFromFile(str(path))
        require(g.intrinsicValue('primitivecount')==0 and not g.vertexAttribs(), 'Only point-only BGEO is supported; primitives/vertices are not silently discarded.')
        this_attrs = []
        this_sig = []
        for a in g.pointAttribs():
            require(not a.isArrayType() and a.dataType() in (hou.attribData.Float, hou.attribData.Int), 'String/array/unsupported point attribute: '+a.name())
            storage = str(a.numericDataType()).split('.')[-1]
            require(storage in code_map, 'Unsupported storage '+storage+'; no implicit precision conversion.')
            size = a.size()
            require(0<size<256, 'Unsupported attribute tuple size.')
            start = sum(v['size'] for v in this_attrs)
            code = code_map[storage]
            this_attrs.append(dict(name=a.name(), size=size, types=code*size, fields=[f'c{i}' for i in range(start,start+size)]))
            this_sig.append((a.name(),size,storage))
        if attrs is None:
            attrs = this_attrs
            signature = this_sig
            types = ''.join(a['types'] for a in attrs)
            dtype = np.dtype([('_begin','u1')]+[(f'c{i}',pc.NUMERIC[c]) for i,c in enumerate(types)]+[('_end','u1')])
        require(this_sig==signature, 'Point attribute schema changes across frames.')
        rows = np.empty(g.intrinsicValue('pointcount'), dtype=dtype)
        rows['_begin']=ord('['); rows['_end']=ord(']')
        for a in attrs:
            method = g.pointIntAttribValues if a['types'][0] in 'bhlq' else g.pointFloatAttribValues
            values = np.asarray(method(a['name'])).reshape(-1,a['size'])
            for col, field in enumerate(a['fields']):
                rows[field]=values[:,col]
        ta = next((a for a in attrs if a['name']==config['time_attribute']), None)
        if ta and len(rows):
            require(ta['size']==1, 'Time must be scalar.')
            ts = rows[ta['fields'][0]]
            require(np.isfinite(ts).all() and (ts==ts[0]).all(), 'Time must be constant and finite within a frame.')
            t = float(ts[0])
        elif g.findGlobalAttrib(config['time_attribute']):
            t = float(g.attribValue(config['time_attribute']))
        elif config['fps'] is not None:
            t = config['time_offset']+(number-config['time_origin_frame'])/config['fps']
        else:
            raise ValueError('No authoritative time for frame '+str(number)+'. Supply explicit --fps/--time-origin-frame/--time-offset; do not infer FPS from names.')
        frames.append(Frame(number,t,rows,path))
        retained_bytes += rows.nbytes
        require(retained_bytes < config['max_memory_mib']*1048576, 'Input rows exceed --max-memory-mib.')
        stamps.append(source_stamp(path))
    return Cache(frames, attrs, dtype, stamps, 'bgeo')


def hbjson_cache(path, config):
    frames=[]; byte_count=0
    with path.open('rb') as f:
        h=pc.read_header(f)
        for _ in range(h.counts['num_frames']):
            fr=pc.read_frame(f,h)
            byte_count += fr.rows.nbytes
            require(byte_count<config['max_memory_mib']*1048576, 'Input rows exceed --max-memory-mib.')
            frames.append(Frame(fr.number,fr.time,fr.rows))
        pc.footer(f)
    require(sum(len(f.rows) for f in frames)==h.counts['num_samples'], 'HBJSON sample count mismatch.')
    return Cache(frames,h.attributes,h.dtype,[source_stamp(path)],'hbjson')


def attr(cache, name):
    matches=[a for a in cache.attrs if a['name']==name]
    require(len(matches)==1, 'Missing or ambiguous attribute: '+name)
    return matches[0]


def values(rows, a):
    return np.column_stack([rows[f] for f in a['fields']])


def validate(cache, config):
    require(len(cache.frames)>0, 'No frames.')
    times=np.array([f.time for f in cache.frames],dtype=np.float64)
    require(np.isfinite(times).all() and (np.diff(times)>0).all(), 'Frame times must be finite and strictly increasing.')
    require(len({f.number for f in cache.frames})==len(cache.frames), 'Duplicate frame labels.')
    a=attr(cache,config['id_attribute'])
    require(a['size']==1 and a['types'] not in ('?','c'), 'ID must be a scalar integer-valued numeric attribute.')
    ids=[]
    for fr in cache.frames:
        ii=pc.ids_for(fr.rows, type('HeaderLike',(),{'attribute':lambda self,n:attr(cache,n)})(), config['id_attribute'])
        require((fr.rows['_begin']==91).all() and (fr.rows['_end']==93).all(), 'Invalid row markers.')
        for aa in cache.attrs:
            for field in aa['fields']:
                require(np.isfinite(fr.rows[field]).all(), 'Non-finite attribute '+aa['name'])
        ids.append(ii)
    union=np.unique(np.concatenate(ids))
    require(len(union)>0, 'Cache has no point IDs.')
    return times,ids,union


def statistics(x):
    if not x:
        return dict(samples=0,median=0.,p95=0.,p99=0.,max=0.)
    a=np.concatenate(x)
    if not len(a): return dict(samples=0,median=0.,p95=0.,p99=0.,max=0.)
    return dict(samples=len(a),median=float(np.median(a)),p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)),max=float(a.max()))


def lifecycle(times,ids,union, masks):
    first=np.full(len(union),-1,dtype=np.int32); last=first.copy()
    kfirst=first.copy(); klast=first.copy(); reappear=0
    for f,ii in enumerate(ids):
        index=np.searchsorted(union,ii)
        prior=last[index]
        reappear+=int(((prior>=0)&(prior!=f-1)).sum())
        first[index[first[index]<0]]=f; last[index]=f
        ix=index[masks[f]]
        kfirst[ix[kfirst[ix]<0]]=f; klast[ix]=f
    has=kfirst>=0
    return dict(original_IDs=len(union),retained_IDs=int(has.sum()),lost_IDs=int((~has).sum()),reappeared_ID_transitions=reappear,
        first_available_sample_changed_IDs=int((has&(first!=kfirst)).sum()),last_available_sample_changed_IDs=int((has&(last!=klast)).sum()),
        max_first_sample_delay_seconds=float((times[kfirst[has]]-times[first[has]]).max()) if has.any() else 0.,
        max_last_sample_advance_seconds=float((times[last[has]]-times[klast[has]]).max()) if has.any() else 0.)


def tolerances(cache, config, matrix, valid, columns):
    result={}
    custom={}
    for spec in config['attribute_tolerance']:
        name,sep,value=spec.partition('=')
        require(sep and name not in custom, 'Use unique --attribute-tolerance NAME=VALUE.')
        v=float(value);require(math.isfinite(v) and v>=0,'Invalid attribute tolerance.')
        custom[name]=v
    for a in cache.attrs:
        n=a['name']
        if n not in columns: continue
        if n in custom:
            require(set(a['types'])<=set('fd'), 'Only float attributes accept interpolation tolerances.')
            result[n]=np.full(valid.shape[1],custom.pop(n))
        elif n==config['position_attribute']:
            result[n]=np.full(valid.shape[1],config['position_tolerance'])
        elif n==config['velocity_attribute']:
            speed=np.linalg.norm(matrix[:,:,columns[n]],axis=2)
            result[n]=np.maximum(config['velocity_tolerance'],config['velocity_relative']*speed.max(axis=0))
        elif n==config['age_attribute'] and set(a['types'])<=set('fd'):
            result[n]=np.full(valid.shape[1],config['age_tolerance'])
        else:
            result[n]=None  # Unknown attributes must be constant across the removed span.
    require(not custom, 'Unknown tolerance attributes: '+','.join(custom))
    return result


def adaptive(cache, config, times, ids, union):
    require(attr(cache,config['position_attribute'])['size']==3,'Adaptive mode requires a 3-component position attribute.')
    columns={}; count=0
    for a in cache.attrs:
        if a['name'] in (config['id_attribute'],config['time_attribute']): continue
        require(set(a['types'])<=set('fd'), 'Adaptive mode currently supports floating point payloads only (integer ID is supported). Use stride for integer/string semantic attributes.')
        columns[a['name']]=list(range(count,count+a['size']));count+=a['size']
    T,N=len(times),len(union)
    estimated=T*N*(8*count+8)+sum(f.rows.nbytes for f in cache.frames)
    require(estimated*2<=config['max_memory_mib']*1048576, f'Adaptive working memory estimate {estimated*2/1048576:.0f} MiB exceeds --max-memory-mib; split batch or increase cap.')
    matrix=np.zeros((T,N,count),dtype=np.float64);valid=np.zeros((T,N),dtype=bool)
    for f,fr in enumerate(cache.frames):
        ii=np.searchsorted(union,ids[f]);valid[f,ii]=True
        for a in cache.attrs:
            if a['name'] in columns:
                matrix[f,ii[:,None],np.array(columns[a['name']])[None,:]]=values(fr.rows,a)
    first=valid.argmax(axis=0);last=T-1-valid[::-1].argmax(axis=0)
    require(np.array_equal(valid.sum(axis=0),last-first+1), 'Adaptive source has ID gaps/reappearances. Use original dense samples or stride; do not infer a lifecycle across gaps.')
    tol=tolerances(cache,config,matrix,valid,columns)
    keep=valid.copy();gap=config['max_gap']
    for iteration in range(config['passes']):
        removed=0
        for f in range(1,T-1):
            ii=np.flatnonzero(keep[f]&(first<f)&(last>f))
            if not len(ii):continue
            prev=f-1-keep[:f,ii][::-1].argmax(axis=0)
            nex=f+1+keep[f+1:,ii].argmax(axis=0)
            ok=(nex-prev)<=gap;ii,prev,nex=ii[ok],prev[ok],nex[ok]
            if not len(ii):continue
            lo=matrix[prev,ii];hi=matrix[nex,ii]
            good=np.ones(len(ii),dtype=bool)
            for j in range(max(1,f-gap+1),min(T-1,f+gap)):
                active=(prev<j)&(nex>j)
                if not active.any():continue
                w=((times[j]-times[prev[active]])/(times[nex[active]]-times[prev[active]]))[:,None]
                prediction=lo[active]+(hi[active]-lo[active])*w
                actual=matrix[j,ii[active]]
                check=np.ones(int(active.sum()),dtype=bool)
                for name,cols in columns.items():
                    if tol[name] is None:
                        check&=(lo[active][:,cols]==hi[active][:,cols]).all(axis=1)&(actual[:,cols]==lo[active][:,cols]).all(axis=1)
                    else:
                        check&=np.linalg.norm(prediction[:,cols]-actual[:,cols],axis=1)<=tol[name][ii[active]]
                good[active]&=check
            keep[f,ii[good]]=False;removed+=int(good.sum())
        print(f'  adaptive pass {iteration+1}: removed {removed}',flush=True)
        if not removed:break
    require(keep[first,np.arange(N)].all() and keep[last,np.arange(N)].all(),'Adaptive dropped an ID boundary.')
    errors={n:[] for n in columns}
    for f in range(1,T-1):
        ii=np.flatnonzero(valid[f]&~keep[f])
        if not len(ii):continue
        prev=f-1-keep[:f,ii][::-1].argmax(axis=0)
        nex=f+1+keep[f+1:,ii].argmax(axis=0)
        require(((nex-prev)<=gap).all(),'Adaptive gap exceeds cap.')
        w=((times[f]-times[prev])/(times[nex]-times[prev]))[:,None]
        lo=matrix[prev,ii];hi=matrix[nex,ii];prediction=lo+(hi-lo)*w;actual=matrix[f,ii]
        for name,cols in columns.items():
            error=np.linalg.norm(prediction[:,cols]-actual[:,cols],axis=1)
            if tol[name] is None:
                require((lo[:,cols]==hi[:,cols]).all() and (actual[:,cols]==lo[:,cols]).all(),'Unknown attribute changed.')
            else:
                require((error<=tol[name][ii]).all(),'Final adaptive reconstruction exceeds tolerance: '+name)
            errors[name].append(error)
    masks=[keep[f,np.searchsorted(union,ii)] for f,ii in enumerate(ids)]
    return masks,dict(attribute_errors={n:statistics(x) for n,x in errors.items()},position_units='raw source position units',unknown_attributes_policy='Constant across span; otherwise keep record',max_source_interval_gap=gap)


def stride(cache,config,times,ids,union):
    kept=set(range(0,len(times),config['stride']))|{len(times)-1}
    required=set(kept)
    first={};last={}
    for f,ii in enumerate(ids):
        for n in ii:
            first.setdefault(int(n),f);last[int(n)]=f
    required.update(first.values());required.update(last.values())
    # Even without boundary preservation, retain a whole original frame for an ID
    # that would otherwise disappear completely, never fabricate its trajectory.
    present=set(int(n) for f in kept for n in ids[f])
    for n in sorted(set(first)-present):
        kept.add(first[n]);present.update(int(i) for i in ids[first[n]])
    if config['preserve_boundaries']:kept.update(required)
    masks=[np.full(len(fr.rows),f in kept,dtype=bool) for f,fr in enumerate(cache.frames)]
    errors={a['name']:[] for a in cache.attrs if a['name'] not in (config['id_attribute'],config['time_attribute']) and set(a['types'])<=set('fd')}
    missing=0
    for f in sorted(set(range(len(times)))-kept):
        left=max(i for i in kept if i<f);right=min(i for i in kept if i>f)
        common,li,ri=np.intersect1d(ids[left],ids[right],return_indices=True)
        _,ci,mi=np.intersect1d(ids[f],common,return_indices=True)
        missing+=len(ids[f])-len(ci)
        w=(times[f]-times[left])/(times[right]-times[left])
        for name in errors:
            a=attr(cache,name)
            lo=values(cache.frames[left].rows,a)[li[mi]].astype(np.float64)
            hi=values(cache.frames[right].rows,a)[ri[mi]].astype(np.float64)
            raw=values(cache.frames[f].rows,a)[ci].astype(np.float64)
            errors[name].append(np.linalg.norm(lo+(hi-lo)*w-raw,axis=1))
    return masks,dict(retained_frame_indices=sorted(kept),attribute_errors={n:statistics(x) for n,x in errors.items()},samples_without_both_ID_neighbors=missing,frames_for_exact_first_last_preservation=len(required),boundary_policy='exact first/last' if config['preserve_boundaries'] else 'report shifts; retain a frame for any otherwise lost ID')


def export_hbjson(path,cache,frames,masks, id_name):
    union=set(int(i) for fr,mask in zip(frames,masks) for i in values(fr.rows[mask],attr(cache,id_name)).ravel())
    counts=dict(num_samples=sum(int(m.sum()) for m in masks),num_frames=len(frames),num_points=len(union))
    with path.open('xb') as f:
        f.write(header_bytes(cache.attrs,counts))
        for fr,mask in zip(frames,masks):
            require(0<=fr.number<2**32,'HBJSON frame number outside uint32.')
            f.write(b'{'+text('number')+struct.pack('<I',fr.number)+text('time')+struct.pack('<f',fr.time)+text('num_points')+struct.pack('<I',int(mask.sum()))+text('frame_data')+b'[')
            f.write(fr.rows[mask].tobytes());f.write(b']}')
        f.write(b']}}')
    with path.open('rb') as f:
        h=pc.read_header(f);require(h.counts==counts,'Re-read header mismatch.')
        for fr,mask in zip(frames,masks):
            got=pc.read_frame(f,h)
            require(got.rows.tobytes()==fr.rows[mask].tobytes() and got.number==fr.number and got.time==float(np.float32(fr.time)), 'HBJSON re-read mismatch.')
        pc.footer(f)


def export_bgeo(folder,cache,frames,masks,mode):
    require(cache.kind=='bgeo','HBJSON -> BGEO is not supported; original geometry metadata is unavailable. Use --format hbjson.')
    folder.mkdir()
    hou=get_hou()
    for fr,mask in zip(frames,masks):
        path=folder/fr.path.name
        if mask.all():
            shutil.copy2(fr.path,path);require(digest(path)==digest(fr.path),'BGEO copy mismatch.')
        else:
            g=hou.Geometry();g.loadFromFile(str(fr.path))
            points=g.points();g.deletePoints(tuple(points[i] for i in np.flatnonzero(~mask)))
            g.saveToFile(str(path))
            check=hou.Geometry();check.loadFromFile(str(path))
            require(check.intrinsicValue('pointcount')==int(mask.sum()),'BGEO count mismatch.')
            for a in cache.attrs:
                method=check.pointIntAttribValues if a['types'][0] in 'bhlq' else check.pointFloatAttribValues
                got=np.asarray(method(a['name'])).reshape(-1,a['size'])
                expected=values(fr.rows[mask],a)
                require(np.array_equal(got,expected),'BGEO re-read attribute mismatch: '+a['name'])


DEFAULTS=dict(mode='inspect',pattern='*.bgeo.sc',format='bgeo',pack=False,stride=2,preserve_boundaries=False,max_points=None,
    id_attribute='id',time_attribute='time',position_attribute='P',velocity_attribute='v',age_attribute='age',
    position_tolerance=0.03,velocity_tolerance=1.0,velocity_relative=0.02,age_tolerance=0.00002,attribute_tolerance=[],max_gap=4,passes=6,
    fps=None,time_origin_frame=1,time_offset=0.,max_memory_mib=2048,max_extract_gib=20.,protect_turns=True)


def run(config):
    require(set(config)<=set(DEFAULTS)|{'source','output'},'Unknown task option: '+str(set(config)-set(DEFAULTS)-{'source','output'}))
    cfg={**DEFAULTS,**config};source=Path(cfg['source']).expanduser().resolve();output=Path(cfg['output']).expanduser().resolve()
    require(source.exists() or glob.has_magic(str(source)),'Input does not exist: '+str(source))
    require(not output.exists(),'Output already exists; choose a new directory: '+str(output))
    require(source!=output and source not in output.parents and output not in source.parents,'Input/output trees must be disjoint.')
    require(cfg['mode'] in ('inspect','stride','adaptive','trim','key_samples'),'Unknown mode.')
    require(cfg['format'] in ('bgeo','hbjson','both'),'Unknown output format.')
    if cfg['mode']=='key_samples':
        require(source.is_file() and source.suffix.lower()=='.hbjson' and cfg['format']=='hbjson',
                'Key sample mode accepts HBJSON input and output only.')
        require(type(cfg['protect_turns']) is bool,'Invalid turning-point option.')
    for key in ('stride','max_gap','passes','max_memory_mib'):
        require(isinstance(cfg[key],int) and not isinstance(cfg[key],bool) and cfg[key]>0,'Invalid positive integer: '+key)
    for key in ('position_tolerance','velocity_tolerance','velocity_relative','age_tolerance','max_extract_gib','time_offset'):
        require(isinstance(cfg[key],(int,float)) and math.isfinite(cfg[key]) and (key=='time_offset' or cfg[key]>=0),'Invalid numeric option: '+key)
    if cfg['fps'] is not None:require(math.isfinite(cfg['fps']) and cfg['fps']>0,'FPS must be positive.')
    if cfg['mode']=='trim':require(isinstance(cfg['max_points'],int) and cfg['max_points']>0,'Trim needs positive --max-points (raw ID exclusive limit).')
    require(isinstance(cfg['attribute_tolerance'],list),'attribute_tolerance must be a list.')
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=output.name+'.part-',dir=output.parent))
    try:
        with tempfile.TemporaryDirectory(prefix='pointcache-input-') as tmp:
            zip_stamp=None
            if source.is_file() and source.suffix.lower()=='.hbjson':
                cache=hbjson_cache(source,cfg)
            else:
                if source.is_file() and source.suffix.lower()=='.zip':
                    zip_stamp=source_stamp(source)
                    paths=safe_extract(source,Path(tmp),cfg['pattern'],cfg['max_extract_gib']*1024**3)
                elif source.is_dir():
                    paths=list(source.rglob(cfg['pattern']))
                elif glob.has_magic(str(source)):
                    paths=[Path(p) for p in glob.glob(str(source))]
                else:paths=[source]
                cache=bgeo_cache(paths,cfg)
            times,ids,union=validate(cache,cfg)
            if cfg['mode']=='adaptive':masks,details=adaptive(cache,cfg,times,ids,union)
            elif cfg['mode']=='stride':masks,details=stride(cache,cfg,times,ids,union)
            elif cfg['mode']=='key_samples':
                import key_samples
                masks,details=key_samples.select(cache,cfg,times,ids,union,sys.modules[__name__])
            elif cfg['mode']=='trim':masks,details=[ii<cfg['max_points'] for ii in ids],{}
            else:masks,details=[np.ones(len(fr.rows),dtype=bool) for fr in cache.frames],{}
            life=lifecycle(times,ids,union,masks)
            if cfg['mode'] in ('adaptive','stride','key_samples'):require(life['lost_IDs']==0,'Temporal reduction lost whole point trajectories.')
            if cfg['mode']=='key_samples':
                require(not life['first_available_sample_changed_IDs'] and not life['last_available_sample_changed_IDs'],
                        'Key sample export changed lifecycle boundaries.')
            selected=details['retained_frame_indices'] if cfg['mode']=='stride' else list(range(len(times)))
            frames=[cache.frames[i] for i in selected]; mm=[masks[i] for i in selected]
            if cfg['mode']!='inspect':
                if cfg['format'] in ('hbjson','both'):export_hbjson(stage/'cache.hbjson',cache,frames,mm,cfg['id_attribute'])
                if cfg['format'] in ('bgeo','both'):export_bgeo(stage/'sequence',cache,frames,mm,cfg['mode'])
            warnings=[]
            if cfg['mode'] in ('adaptive','key_samples'):warnings.append('Sparse per-ID sample sequence. A missing point at a frame is not a death event. Requires time/ID interpolation; ordinary per-frame geometry playback will omit points.')
            if cfg['mode']=='stride':warnings.append('Whole-frame thinning is lossy. Preserve original frame/time spacing; missing files are not automatically interpolated by File SOP.')
            if life['first_available_sample_changed_IDs'] or life['last_available_sample_changed_IDs']:warnings.append('First/last available sample times changed; see lifecycle report. Authored age/life in retained records are unchanged.')
            if cfg['mode']=='trim':warnings.append('Only raw IDs 0 <= id < max_points are retained. This is not a uniform spatial or temporal subset.')
            if times[0]!=0.:warnings.append('Original time does not start at zero; no startup samples were fabricated.')
            outputs=[dict(path=p.relative_to(stage).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(stage.rglob('*')) if p.is_file()]
            for stamp in cache.sources:require(digest(stamp['path'])==stamp['sha256'],'Input geometry changed during processing.')
            if zip_stamp:require(digest(source)==zip_stamp['sha256'],'Input ZIP changed during processing.')
            report=dict(tool_version=VERSION,validation_passed=True,mode=cfg['mode'],config=cfg,source_kind=cache.kind,source_container=zip_stamp,hbjson_frame_time_encoding='float32 (format-defined); point time attribute retains its original type/values',
                source_files=cache.sources if not zip_stamp else [{**s,'path':Path(s['path']).name} for s in cache.sources],
                attributes=cache.attrs,source_frames=len(times),output_frames=len(frames),source_samples=sum(len(f.rows) for f in cache.frames),output_samples=sum(int(m.sum()) for m in mm),
                original_time_range=times[[0,-1]].tolist(),output_time_range=[frames[0].time,frames[-1].time],lifecycle=life,analysis=details,
                frames=[dict(number=fr.number,time=fr.time,source_points=len(fr.rows),retained_samples=int(m.sum())) for fr,m in zip(frames,mm)],outputs=outputs,
                source_timeline=[dict(number=fr.number,time=fr.time,source_points=len(fr.rows)) for fr in cache.frames],
                original_sources_unchanged=True,retained_attribute_values_unchanged=True,UE_modified=False,UE_imported=False,GPU_compiled=False,visual_validation=False,warnings=warnings)
            write_json(stage/'validation.json',report)
            with (stage/'README.txt').open('x',encoding='utf-8-sig') as f:
                f.write('Houdini point-cache batch toolkit '+VERSION+'\nMode: '+cfg['mode']+'\n')
                f.write(f'Frames: {len(times)} -> {len(frames)}; samples: {report["source_samples"]} -> {report["output_samples"]}\n')
                f.write(f'Time: {times[0]} -> {times[-1]} seconds, unchanged; stable IDs not renumbered.\n')
                f.write('\n'.join(warnings)+'\nNo UE/GPU/visual validation. See validation.json for hashes, errors and lifecycle coverage.\n')
            package=None
            if cfg['pack']:
                package=stage/'result.zip'
                members=[p for p in stage.rglob('*') if p.is_file() and p!=package]
                with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
                    for p in sorted(members):z.write(p,arcname=p.relative_to(stage).as_posix())
                with zipfile.ZipFile(package) as z:
                    require(z.testzip() is None,'ZIP CRC failed.')
                    for p in members:require(hashlib.sha256(z.read(p.relative_to(stage).as_posix())).hexdigest()==digest(p),'ZIP payload mismatch.')
            result=dict(output=str(output),mode=cfg['mode'],source_frames=len(times),output_frames=len(frames),source_samples=report['source_samples'],output_samples=report['output_samples'],IDs=life['retained_IDs'],report=str(output/'validation.json'))
            if package:result.update(zip=str(output/'result.zip'),zip_bytes=package.stat().st_size,zip_sha256=digest(package))
            stage.rename(output)
            return result
    except Exception as e:
        raise ValueError(str(e)+f' [Unpublished partial output: {stage}]') from e


def add_options(parser):
    parser.add_argument('--mode',choices=['inspect','stride','adaptive','trim'],default=argparse.SUPPRESS)
    parser.add_argument('--format',choices=['bgeo','hbjson','both'],default=argparse.SUPPRESS)
    parser.add_argument('--pack',action='store_true',default=argparse.SUPPRESS)
    parser.add_argument('--preserve-boundaries',action='store_true',default=argparse.SUPPRESS)
    for name in ('source','output','pattern','id-attribute','time-attribute','position-attribute','velocity-attribute','age-attribute'):
        parser.add_argument('--'+name,default=argparse.SUPPRESS)
    for name in ('stride','max-points','max-gap','passes','max-memory-mib','time-origin-frame'):
        parser.add_argument('--'+name,type=int,default=argparse.SUPPRESS)
    for name in ('fps','time-offset','position-tolerance','velocity-tolerance','velocity-relative','age-tolerance','max-extract-gib'):
        parser.add_argument('--'+name,type=float,default=argparse.SUPPRESS)
    parser.add_argument('--attribute-tolerance',action='append',default=argparse.SUPPRESS)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    single=sub.add_parser('run');add_options(single)
    batch=sub.add_parser('batch');batch.add_argument('manifest',type=Path);batch.add_argument('--output-root',type=Path,required=True)
    args=vars(p.parse_args(argv))
    if args.pop('command')=='run':
        try:
            require('source' in args and 'output' in args,'run needs --source and --output.')
            print(json.dumps(run(args),ensure_ascii=False,indent=2));return 0
        except Exception as e:
            print('ERROR: '+str(e),file=sys.stderr);return 2
    root=args['output_root'].resolve(); manifest=args['manifest'].resolve()
    try:
        require(not root.exists(),'Batch output root exists; choose a new directory.')
        jobs=json.loads(manifest.read_text(encoding='utf-8-sig'))
        require(set(jobs)<= {'defaults','jobs'} and isinstance(jobs.get('jobs'),list),'Expected {defaults: {...}, jobs: [...]} manifest.')
        names=[]
        for job in jobs['jobs']:
            name=job.get('name','')
            require(re.fullmatch(r'[A-Za-z0-9_-]{1,80}',name) is not None,'Job name needs 1-80 ASCII letters/digits/_/-.')
            require(name.upper().split('.')[0] not in {'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]},'Reserved job name.')
            require(name.casefold() not in names,'Duplicate job name.');names.append(name.casefold())
        root.mkdir(parents=True)
        results=[]
        for job in jobs['jobs']:
            config={**jobs.get('defaults',{}),**{k:v for k,v in job.items() if k!='name'}}
            name=job['name']
            try:
                require('source' in config and 'output' not in config,'Each job needs source; output is assigned by output-root/name.')
                source=Path(config['source']).expanduser()
                if not source.is_absolute():source=manifest.parent/source
                config.update(source=str(source),output=str(root/name))
                print('JOB '+name,flush=True)
                results.append(dict(name=name,success=True,result=run(config)))
            except Exception as e:
                results.append(dict(name=name,success=False,error=str(e)));print('FAILED '+name+': '+str(e),flush=True)
        report=dict(tool_version=VERSION,manifest=str(manifest),jobs=results,succeeded=sum(r['success'] for r in results),failed=sum(not r['success'] for r in results))
        write_json(root/'batch-report.json',report)
        print(json.dumps(report,ensure_ascii=False,indent=2));return 1 if report['failed'] else 0
    except Exception as e:
        print('ERROR: '+str(e),file=sys.stderr);return 2


if __name__=='__main__':
    raise SystemExit(main())
