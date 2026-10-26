"""CPU occurrence-ID membership and cyclic scanning in the original fixed P.

Importing this helper does not change process environment, CPU threads, or RNG.
It never touches model/optimizer/CUDA state or enumerates reconstruction candidates.
"""
import fcntl,hashlib,json,os,time,uuid
from pathlib import Path
import numpy as np

RANDOM80_SEED=42 ^ 0xC0FFEE
EXPECTED_N={'beauty':95297910,'toys':92445461}

def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def _atomic(path,value):
    tmp=path.with_name(path.name+'.pending')
    with tmp.open('x') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def _binding(shuffle_assets_path,dataset):
    root=Path(shuffle_assets_path).resolve();source=json.loads((root/'manifest.json').read_text())
    n=source['total_occurrences']
    if source.get('ready') is not True or n!=EXPECTED_N[dataset]:raise ValueError('Expected completed literal P1 shuffle assets')
    permutation=root/'permutation.npy';expected=source['files']['permutation.npy']['sha256']
    if file_sha256(permutation)!=expected:raise ValueError('Original fixed permutation checksum changed')
    return root,source,dict(dataset=dataset,selection='random80',method='random80',source_stream_id=source['source_stream_id'],
        source_asset_identity=source['immutable_identity'],source_manifest_sha256=file_sha256(root/'manifest.json'),
        replay_shuffle_asset_identity=source['immutable_identity'],
        permutation_sha256=expected,total_occurrences=n,selected_count=4*n//5,seed=RANDOM80_SEED,
        algorithm='Private torch.randperm over occurrence IDs; retain first floor(4N/5); map membership to sorted original P positions')

def load_selected_positions(selection_dir,shuffle_assets_path):
    """Return (read-only positions mmap, manifest), with source/hash/order guards."""
    root=Path(selection_dir).resolve();manifest=root/'manifest.json'
    if not manifest.exists():manifest=root/'selected_manifest.json'
    m=json.loads(manifest.read_text())
    if 'total_occurrences' not in m:m['total_occurrences']=m['occurrence_count']
    source=json.loads((Path(shuffle_assets_path)/'manifest.json').read_text())
    if m.get('ready') is not True or m['source_stream_id']!=source['source_stream_id'] or m.get('replay_shuffle_asset_identity',m.get('source_asset_identity'))!=source['immutable_identity'] or m['permutation_sha256']!=source['files']['permutation.npy']['sha256']:
        raise ValueError('Selection does not belong to these original fixed-shuffle assets')
    if m['total_occurrences']!=source['total_occurrences'] or m['selected_count']!=4*m['total_occurrences']//5:
        raise ValueError('Selection is not exact floor(4N/5) of this occurrence population')
    if m['source_manifest_sha256']!=file_sha256(Path(shuffle_assets_path)/'manifest.json'):
        raise ValueError('Source shuffle manifest bytes changed')
    path=root/'selected_positions.npy'
    if file_sha256(path)!=m['files']['selected_positions.npy']['sha256']:raise ValueError('Selection positions checksum mismatch')
    positions=np.load(path,mmap_mode='r',allow_pickle=False);n=m['total_occurrences']
    if positions.dtype!=np.int64 or positions.ndim!=1 or len(positions)!=m['selected_count'] or not len(positions):raise ValueError('Invalid positions dtype/shape/count')
    if int(positions[0])<0 or int(positions[-1])>=n:raise ValueError('Selected P position outside original permutation')
    previous=-1
    for start in range(0,len(positions),1000000):
        block=positions[start:start+1000000]
        if int(block[0])<=previous or np.any(block[1:]<=block[:-1]):raise ValueError('Positions must be sorted unique original P coordinates')
        previous=int(block[-1])
    return positions,m

def ensure_selection(dm):
    """Bind a prepared random selection or prepare it once behind the CPU lock.

    Caller invokes this on all ranks at the same post200 boundary; no model,
    optimizer, RNG restoration or input materialization is performed here.
    """
    prepared=os.environ.get('REPLAY_SELECTION_PREPARED')
    output=Path(prepared).resolve() if prepared else Path(dm.selection_output_dir).resolve()
    if prepared:
        load_selected_positions(output,dm.assets_path)
    else:
        dataset=dm.asset_manifest.get('identity',{}).get('dataset',Path(dm.assets_path).name)
        if dm.trainer.is_global_zero:prepare_random80(dm.assets_path,output,dataset)
    dm.trainer.strategy.barrier('random80_selection_ready')
    load_selected_positions(output,dm.assets_path)
    return output

ensure_random_selection=ensure_selection

def scan_selected_positions(positions,cursor,k,n):
    """Take K retained entries at/after absolute original-P cursor, wrapping P.

    Return P positions in [0,N) and next absolute cursor. Checkpoint this cursor,
    not a fabricated selected-index offset; repeated tuples remain distinct IDs.
    """
    cursor=int(cursor);k=int(k);n=int(n)
    if cursor<0 or k<0 or n<=0 or not len(positions):raise ValueError('Invalid scan cursor/K/N or empty selection')
    if k==0:return np.empty(0,dtype=np.int64),cursor
    cycle,within=divmod(cursor,n);index=int(np.searchsorted(positions,within,side='left'))
    if index==len(positions):cycle+=1;index=0
    logical=index+np.arange(k,dtype=np.int64)
    selected=np.asarray(positions[logical%len(positions)],dtype=np.int64)
    final_cycle=cycle+int(logical[-1]//len(positions))
    return selected,final_cycle*n+int(selected[-1])+1

def scan_retained_occurrences(permutation,positions,cursor,k):
    selected,next_cursor=scan_selected_positions(positions,cursor,k,len(permutation))
    return np.asarray(permutation[selected],dtype=np.int64),next_cursor

def prepare_random80(shuffle_assets_path,selection_dir,dataset):
    """Prepare/reuse immutable CPU assets; no GPU launch or gate publication."""
    import torch
    started=time.perf_counter();cpu=time.process_time();root,source,binding=_binding(shuffle_assets_path,dataset)
    binding['torch_version']=str(torch.__version__)
    identity=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    target=Path(selection_dir).resolve();target.parent.mkdir(parents=True,exist_ok=True)
    with (target.parent/'.random80.prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (target/'manifest.json').exists():
            positions,m=load_selected_positions(target,root)
            if m['immutable_identity']!=identity:raise ValueError('Existing ready selection has a different seed/source identity; refusing overwrite')
            return dict(manifest=m,manifest_path=str(target/'manifest.json'),manifest_sha256=file_sha256(target/'manifest.json'),reused=True,
                        cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-started)
        if target.exists():raise FileExistsError('Unpublished selection exists; preserve and inspect it')
        stage=target.parent/f'.random80.pending_{uuid.uuid4().hex}';stage.mkdir();n=binding['total_occurrences'];count=binding['selected_count']
        p=np.load(root/'permutation.npy',mmap_mode='r',allow_pickle=False)
        if p.dtype!=np.int64 or p.shape!=(n,):raise ValueError('Original P dtype/shape differs')
        rng=torch.get_rng_state().clone();generator=torch.Generator(device='cpu').manual_seed(RANDOM80_SEED)
        order=torch.randperm(n,generator=generator,dtype=torch.int64,device='cpu')
        if not torch.equal(rng,torch.get_rng_state()):raise AssertionError('Random selection changed global model RNG')
        membership=np.zeros(n,dtype=np.bool_);membership[order[:count].numpy()]=True;del order
        if int(membership.sum())!=count:raise AssertionError('Retained occurrence IDs are not unique/count-exact')
        positions=np.lib.format.open_memmap(stage/'selected_positions.npy',mode='w+',dtype=np.int64,shape=(count,))
        written=0
        for start in range(0,n,1000000):
            block=np.asarray(p[start:start+1000000]);found=np.flatnonzero(membership[block])+start
            positions[written:written+len(found)]=found;written+=len(found)
        if written!=count:raise AssertionError('Original P membership projection lost/duplicated an occurrence ID')
        previous=-1
        for start in range(0,count,1000000):
            block=positions[start:start+1000000]
            if int(block[0])<=previous or np.any(block[1:]<=block[:-1]) or not bool(membership[p[block]].all()):raise AssertionError('Sorted positions/membership verification failed')
            previous=int(block[-1])
        if int(positions[0])<0 or int(positions[-1])>=n or int(positions[-1])==count-1:raise AssertionError('Selection is invalid or exactly original P initial80% prefix')
        positions.flush();del membership
        tests=[]
        for cursor,k in [(0,7),(n-2,11),(n+5,8),(2*n-1,3)]:
            chosen,next_cursor=scan_selected_positions(positions,cursor,k,n)
            first,mid=scan_selected_positions(positions,cursor,k//2,n)
            second,end=scan_selected_positions(positions,mid,k-k//2,n)
            if not np.array_equal(chosen,np.concatenate([first,second])) or next_cursor!=end:raise AssertionError('Split/resume/cycle scan differs')
            tests.append(dict(cursor=cursor,k=k,passed=True))
        path=stage/'selected_positions.npy';files={'selected_positions.npy':dict(path=str(target/path.name),sha256=file_sha256(path),dtype='int64',shape=[count],bytes=path.stat().st_size)}
        m=dict(format_version=1,ready=True,**binding,immutable_identity=identity,files=files,
               verification=dict(exact_count=True,unique_sorted_in_range=True,independent_private_rng_unchanged=True,not_initial_P80_prefix=True,
                   membership_of_occurrence_IDs=True,split_resume_wrap=tests,no_candidate_enumeration_or_dedup=True),
               cost=dict(cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-started,gpu_seconds=0),created_unix=time.time())
        del positions,p;os.rename(stage,target);_atomic(target/'manifest.json',m)
        return dict(manifest=m,manifest_path=str(target/'manifest.json'),manifest_sha256=file_sha256(target/'manifest.json'),reused=False,
                    cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-started)
