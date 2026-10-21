#!/usr/bin/env python3
"""Recompute an RNA-expression embedding on every accepted scTCR-covered T cell."""
from release_config import release_path, private_value
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '16'
import argparse, gc, gzip, hashlib, json, platform, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.io import mmread
import scipy.io._fast_matrix_market as fast_matrix_market
from scipy import sparse
import anndata as ad
import scanpy as sc
import harmonypy
from threadpoolctl import threadpool_limits
from importlib.metadata import version
fast_matrix_market.PARALLELISM = 16
if hasattr(os, 'sched_setaffinity'):
    cpu_affinity = sorted(os.sched_getaffinity(0))[:16]
    for native_thread in Path(f'/proc/{os.getpid()}/task').iterdir():
        os.sched_setaffinity(int(native_thread.name), cpu_affinity)

BASE = Path(str(release_path('recompute_embedding_source_25')))
OUT = BASE / 'recomputed_scTCR_20261009'
ANA = OUT / 'analysis'
SOURCE = Path(str(release_path('recompute_embedding_source_28')))
SEED = 20261009
MARKERS = ['CD3D','CD3E','CD4','CD8A','CD8B','CCR7','SELL','TCF7','FOXP3','IL2RA','GZMK','CX3CR1','GZMB','PDCD1','LAG3','HAVCR2','CXCL13','LAYN','TRDV2','KLRB1','SLC4A10','MKI67','TRDC','NKG7','GNLY','IFNG','IL7R','CTLA4','TOX','FOS','JUN','HSPA1A']

def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n')

def file_stats(path):
    s = path.stat()
    return {'path': str(path), 'size_bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}

def digest_lines(values):
    return hashlib.sha256(('\n'.join(values)+'\n').encode()).hexdigest()

def extract():
    target = ANA / 'selected_counts.h5ad'
    if target.exists():
        log('Reusing completed selected_counts.h5ad checkpoint')
        return
    frame = pd.read_csv(BASE / 'GSE236581_scRNA_scTCR_Tcells_subtypes_UMAP.csv')
    assert len(frame)==369063 and frame.cell_name.is_unique
    assert (frame.has_accepted_alpha | frame.has_accepted_beta).all()
    assert frame.MajorCellType.eq('T').all()
    barcodes = pd.read_csv(SOURCE/'GSE236581_barcodes.tsv.gz',header=None,sep='\t')[0].astype(str)
    features = pd.read_csv(SOURCE/'GSE236581_features.tsv.gz',header=None,sep='\t',names=['gene_id','gene_symbol','feature_type'])
    assert barcodes.is_unique
    positions = pd.Index(barcodes).get_indexer(frame.cell_name)
    assert (positions>=0).all() and len(np.unique(positions))==len(frame)
    paths=[BASE/'GSE236581_scRNA_scTCR_Tcells_subtypes_UMAP.csv', SOURCE/'GSE236581_counts.mtx.gz',SOURCE/'GSE236581_barcodes.tsv.gz',SOURCE/'GSE236581_features.tsv.gz']
    provenance={'sources':[file_stats(p) for p in paths], 'source_barcode_count':len(barcodes),'source_feature_count':len(features),'selected_cell_count':len(frame),'selected_cell_names_unique':bool(frame.cell_name.is_unique),'selected_cells_all_matched_to_count_barcodes':True,'selected_cell_order_sha256':digest_lines(frame.cell_name.tolist()),'barcode_order_sha256':digest_lines(barcodes.tolist()),'selection':'Exact existing CSV cell_name set; accepted TRA or TRB; all source tissue/treatment/cancer labels retained; no visual outlier filtering','excluded_cells':0}
    write_json(ANA/'input_provenance.json',provenance)
    log('Loading complete MatrixMarket with SciPy fast reader; 36,027 genes x 975,275 cells')
    with threadpool_limits(limits=16):
        counts=mmread(SOURCE/'GSE236581_counts.mtx.gz')
    assert counts.shape==(36027,975275) and counts.nnz==1310816895
    log(f'Parsed matrix nnz={counts.nnz}; converting CSC for exact barcode selection')
    counts=counts.tocsc()
    subset=counts[:,positions].T.tocsr().astype(np.float32)
    del counts; gc.collect()
    subset.sum_duplicates(); subset.sort_indices()
    assert (subset.data >= 0).all() and np.equal(subset.data,np.floor(subset.data)).all()
    obs=frame.drop(columns=['UMAP1','UMAP2','coordinate_source','draw_order','point_alpha']).set_index('cell_name')
    for col in obs.select_dtypes(include=['object']).columns:
        obs[col]=obs[col].astype(str).astype('category')
    features.index=pd.Index(features.gene_symbol.astype(str),name='feature_name')
    obj=ad.AnnData(X=subset,obs=obs,var=features)
    obj.var_names_make_unique()
    obj.var['mt']=obj.var.gene_symbol.str.upper().str.startswith('MT-').to_numpy()
    sc.pp.calculate_qc_metrics(obj,qc_vars=['mt'],percent_top=None,log1p=False,inplace=True)
    assert (obj.obs.total_counts>0).all()
    obj.uns['selection_note']=provenance['selection']
    log(f'Writing selected raw count checkpoint: {obj.shape}, nnz={obj.X.nnz}')
    obj.write_h5ad(str(target)+'.partial',compression='lzf')
    os.replace(str(target)+'.partial',target)
    obj.obs[['Ident','Patient','SubCellType','Tissue','total_counts','n_genes_by_counts','pct_counts_mt']].to_csv(ANA/'cell_input_qc.tsv.gz',sep='\t')
    provenance.update({'selected_count_nnz':int(obj.X.nnz),'selected_matrix_shape':list(obj.shape),'zero_count_cells':int((obj.obs.total_counts==0).sum()),'subtype_count':int(obj.obs.SubCellType.nunique()),'sample_count':int(obj.obs.Ident.nunique()),'patient_count':int(obj.obs.Patient.nunique())})
    write_json(ANA/'input_provenance.json',provenance)
    log('Raw count extraction checkpoint COMPLETE')

def save_checkpoint(obj, name):
    target=ANA/name
    obj.write_h5ad(str(target)+'.partial',compression='lzf')
    os.replace(str(target)+'.partial',target)

def prepare_pca():
    path=ANA/'pca_checkpoint.h5ad'
    if path.exists():
        log('Loading PCA checkpoint')
        return sc.read_h5ad(path)
    log('Loading selected count checkpoint and selecting 3000 batch-aware HVGs')
    obj=sc.read_h5ad(ANA/'selected_counts.h5ad')
    receptor=obj.var.gene_symbol.str.match(r'^TR[ABGD][VDJ]').to_numpy()
    eligible=(obj.var.n_cells_by_counts.to_numpy()>=3) & ~receptor
    work=obj[:,eligible].copy()
    sample_sizes=work.obs.Ident.value_counts()
    small_samples=sample_sizes.index[sample_sizes<30].astype(str).tolist()
    work.obs['hvg_batch']=work.obs.Ident.astype(str)
    work.obs.loc[work.obs.Ident.astype(str).isin(small_samples),'hvg_batch']='__small_samples_lt30__'
    work.obs['hvg_batch']=work.obs['hvg_batch'].astype('category')
    grouping=work.obs[['Ident','hvg_batch']].drop_duplicates().copy()
    grouping['n_cells_original_sample']=grouping.Ident.map(sample_sizes).astype(int)
    grouping.to_csv(ANA/'hvg_effective_batch_grouping.tsv',sep='\t',index=False)
    adaptation={'initial_failure':"seurat_v3 by original Ident, span=0.3: ValueError There are other near singularities as well. 0.090619",'strategy':'HVG fitting only: pool samples with fewer than 30 selected T cells; retain original Ident for Harmony','pooled_samples':small_samples,'pooled_total_cells':int(sample_sizes.loc[small_samples].sum()),'original_batch_count':int(work.obs.Ident.nunique()),'effective_batch_count':int(work.obs.hvg_batch.nunique()),'attempts':[]}
    for strategy in ['pooled_small_samples','learn_samples_ge30']:
        learning=work if strategy=='pooled_small_samples' else work[~work.obs.Ident.astype(str).isin(small_samples)].copy()
        batch_key='hvg_batch' if strategy=='pooled_small_samples' else 'Ident'
        log(f'HVG seurat_v3 span=0.3; strategy={strategy}, fitting_cells={learning.n_obs}')
        try:
            sc.pp.highly_variable_genes(learning,flavor='seurat_v3',n_top_genes=3000,batch_key=batch_key,span=0.3)
        except ValueError as error:
            if not any(term in str(error).lower() for term in ['singular','extrapolation','reciprocal condition']): raise
            adaptation['attempts'].append({'strategy':strategy,'n_learning_cells':learning.n_obs,'span':0.3,'status':'failed','error':str(error)})
            write_json(ANA/'hvg_adaptation.json',adaptation)
            log(f'LOESS failed with {strategy}: {error}; proceeding to documented robust feature-learning fallback')
        else:
            work.var=learning.var.copy()
            adaptation['attempts'].append({'strategy':strategy,'n_learning_cells':learning.n_obs,'span':0.3,'status':'success'})
            adaptation['effective_span']=0.3
            adaptation['effective_batch_key']=batch_key
            adaptation['n_learning_cells']=learning.n_obs
            adaptation['final_strategy']=strategy
            adaptation['effective_flavor']='seurat_v3'
            write_json(ANA/'hvg_adaptation.json',adaptation)
            break
    else:
        log('LOESS remained singular; using batch-aware normalized-dispersion HVG fallback, all cells retained')
        scales=1e4/learning.obs.total_counts.to_numpy()
        learning.X=sparse.diags(scales)@learning.X
        sc.pp.log1p(learning)
        sc.pp.highly_variable_genes(learning,flavor='seurat',n_top_genes=3000,batch_key='Ident')
        work.var=learning.var.copy()
        adaptation['effective_span']=None; adaptation['effective_batch_key']='Ident'
        adaptation['effective_flavor']='seurat'; adaptation['n_learning_cells']=learning.n_obs
        adaptation['final_strategy']='Normalized-dispersion batch-aware HVG on samples with >=30 selected cells, exact full counts normalization denominator'
        write_json(ANA/'hvg_adaptation.json',adaptation)
    if learning is not work: del learning
    gc.collect()
    grouping['included_in_hvg_learning']=True if adaptation['final_strategy']=='pooled_small_samples' else ~grouping.Ident.astype(str).isin(small_samples)
    grouping['effective_hvg_group']=grouping.hvg_batch.astype(str) if adaptation['final_strategy']=='pooled_small_samples' else grouping.Ident.astype(str)
    grouping.loc[~grouping.included_in_hvg_learning,'effective_hvg_group']='__not_used_for_feature_learning__'
    grouping=grouping.rename(columns={'hvg_batch':'attempted_pool_group'})
    grouping.to_csv(ANA/'hvg_effective_batch_grouping.tsv',sep='\t',index=False)
    adaptation['attempted_pool_effective_batch_count']=adaptation.pop('effective_batch_count')
    adaptation['attempted_pool_strategy']=adaptation.pop('strategy')
    adaptation['effective_batch_count']=int(grouping.loc[grouping.included_in_hvg_learning,'effective_hvg_group'].nunique())
    adaptation['n_cells_excluded_from_hvg_learning_only']=int(grouping.loc[~grouping.included_in_hvg_learning,'n_cells_original_sample'].sum())
    adaptation['n_cells_in_all_downstream_analysis']=obj.n_obs
    adaptation['effective_grouping_file']='hvg_effective_batch_grouping.tsv'
    write_json(ANA/'hvg_adaptation.json',adaptation)
    parameters=json.loads((ANA/'parameters.json').read_text())
    parameters['hvg']['batch_key']=adaptation['effective_batch_key']
    parameters['hvg']['span']=adaptation['effective_span']
    parameters['hvg']['flavor']=adaptation['effective_flavor']
    parameters['hvg']['n_learning_cells']=adaptation['n_learning_cells']
    parameters['hvg']['final_strategy']=adaptation['final_strategy']
    parameters['hvg']['effective_batch_count']=adaptation['effective_batch_count']
    parameters['hvg']['n_cells_excluded_from_feature_learning_only']=adaptation['n_cells_excluded_from_hvg_learning_only']
    parameters['hvg']['adaptation_file']='hvg_adaptation.json'
    parameters['hvg']['effective_grouping_file']='hvg_effective_batch_grouping.tsv'
    write_json(ANA/'parameters.json',parameters)
    hvg_names=work.var_names[work.var.highly_variable].tolist()
    assert len(hvg_names)==3000
    work.var.to_csv(ANA/'hvg_statistics.tsv.gz',sep='\t')
    obj.var.loc[receptor,['gene_id','gene_symbol','n_cells_by_counts']].to_csv(ANA/'TCR_variable_diversity_joining_genes_excluded_from_HVG.tsv',sep='\t')
    del work; gc.collect()
    log('Normalizing every cell to 10000 total counts and log1p; retaining marker expression')
    sc.pp.normalize_total(obj,target_sum=1e4)
    sc.pp.log1p(obj)
    present=[m for m in MARKERS if m in obj.var_names]
    missing=[m for m in MARKERS if m not in obj.var_names]
    sparse.save_npz(ANA/'marker_log1p_normalized.npz',obj[:,present].X.tocsr())
    write_json(ANA/'marker_genes.json',{'genes':present,'missing':missing,'expression':'ln(1 + 10000 * raw counts / total raw counts per cell)','cell_order_sha256':digest_lines(obj.obs_names.tolist())})
    obj=obj[:,hvg_names].copy()
    obj.var['highly_variable']=True
    normalized=obj.X.copy()
    log('Scaling HVGs with variance standardization, zero centering and clipping at 10')
    sc.pp.scale(obj,zero_center=True,max_value=10)
    log('Computing 50-component randomized PCA from the new scTCR-covered subset')
    sc.pp.pca(obj,n_comps=50,svd_solver='randomized',random_state=SEED)
    obj.X=normalized
    ratios=obj.uns['pca']['variance_ratio']
    pd.DataFrame({'PC':np.arange(1,len(ratios)+1),'variance_ratio':ratios,'cumulative_variance_ratio':np.cumsum(ratios)}).to_csv(ANA/'pca_variance.tsv',sep='\t',index=False)
    log(f'PCA variance: first30={ratios[:30].sum():.4f}; first50={ratios.sum():.4f}')
    save_checkpoint(obj,'pca_checkpoint.h5ad')
    return obj

def correct_and_graph(obj):
    graph=ANA/'graph_checkpoint.h5ad'
    if graph.exists():
        log('Loading graph checkpoint')
        return sc.read_h5ad(graph)
    harmony_path=ANA/'harmony_pca.npy'
    if harmony_path.exists():
        obj.obsm['X_pca_harmony']=np.load(harmony_path)
    else:
        log('Running Harmony by Ident (sample), all 50 PCs; theta=2, lambda=1, 16 threads')
        result=harmonypy.run_harmony(obj.obsm['X_pca'],obj.obs,'Ident',theta=2,lamb=1,nclust=100,max_iter_harmony=20,max_iter_kmeans=10,random_state=SEED,ncores=16,verbose=True)
        corrected=np.asarray(result.Z_corr,dtype=np.float32)
        if corrected.shape != obj.obsm['X_pca'].shape:
            corrected=corrected.T
        assert corrected.shape==obj.obsm['X_pca'].shape and np.isfinite(corrected).all()
        obj.obsm['X_pca_harmony']=corrected
        np.save(harmony_path,corrected)
        diagnostic={}
        for name in ['objective_harmony','objective_kmeans','kmeans_rounds']:
            if hasattr(result,name): diagnostic[name]=np.asarray(getattr(result,name)).tolist()
        write_json(ANA/'harmony_diagnostics.json',diagnostic)
        del result; gc.collect()
    log('Building 30-neighbor graph on first 30 newly computed Harmony PCs')
    sc.pp.neighbors(obj,n_neighbors=30,n_pcs=30,use_rep='X_pca_harmony',metric='euclidean',method='umap',transformer='pynndescent',random_state=SEED)
    for res in [0.4,0.8,1.2]:
        key=f'leiden_r{res:.1f}'.replace('.','_')
        log(f'Clustering {key}, igraph backend, 2 iterations, undirected')
        sc.tl.leiden(obj,resolution=res,key_added=key,flavor='igraph',n_iterations=2,directed=False,random_state=SEED)
        log(f'{key}: {obj.obs[key].nunique()} clusters')
    save_checkpoint(obj,'graph_checkpoint.h5ad')
    return obj

def summarize(obj):
    log('Writing coordinates, cluster composition, expression and graph quality summaries')
    coords=pd.DataFrame(obj.obsm['X_umap'],index=obj.obs_names,columns=['UMAP1','UMAP2'])
    keys=['leiden_r0_4','leiden_r0_8','leiden_r1_2']
    for key in keys: coords[key]=obj.obs[key].astype(str).to_numpy()
    coords.index.name='cell_name'
    coords.to_csv(ANA/'coordinates.tsv.gz',sep='\t')
    frames=[]
    for key in keys:
        count=pd.crosstab(obj.obs[key],obj.obs.SubCellType)
        frame=count.stack().rename('n_cells').reset_index().rename(columns={key:'cluster'})
        frame['cluster']=frame['cluster'].astype(str)
        frame['SubCellType']=frame['SubCellType'].astype(str)
        frame['resolution']=key
        frame['fraction_of_cluster']=frame.n_cells/frame.cluster.map(count.sum(axis=1))
        frame['fraction_of_subtype']=frame.n_cells/frame.SubCellType.map(count.sum(axis=0))
        frames.append(frame)
    pd.concat(frames,ignore_index=True).to_csv(ANA/'cluster_subtype_composition.tsv',sep='\t',index=False)
    composition=[]
    cluster_qc=[]
    for key in keys:
        for label in ['Ident','Patient','Tissue','treatment_stage','Cancer Type']:
            count=pd.crosstab(obj.obs[key],obj.obs[label])
            frame=count.stack().rename('n_cells').reset_index().rename(columns={key:'cluster',label:'level'})
            frame['cluster']=frame['cluster'].astype(str)
            frame['resolution']=key; frame['covariate']=label
            frame['fraction_of_cluster']=frame.n_cells/frame.cluster.map(count.sum(axis=1))
            composition.append(frame)
        for cluster,group in obj.obs.groupby(key,observed=True):
            sample=group.Ident.value_counts(); sample=sample[sample>0]
            prob=sample.to_numpy()/len(group)
            subtype=group.SubCellType.value_counts()
            cluster_qc.append({'resolution':key,'cluster':cluster,'n_cells':len(group),'n_samples':len(sample),'n_patients':group.Patient.nunique(),'dominant_sample':str(sample.index[0]),'dominant_sample_fraction':float(prob.max()),'sample_entropy_nats':float(-(prob*np.log(prob)).sum()),'effective_sample_count':float(np.exp(-(prob*np.log(prob)).sum())),'dominant_author_subtype':str(subtype.index[0]),'dominant_author_subtype_fraction':float(subtype.iloc[0]/len(group)),'median_total_counts':float(group.total_counts.median()),'median_n_genes':float(group.n_genes_by_counts.median()),'median_pct_counts_mt':float(group.pct_counts_mt.median())})
    pd.concat(composition,ignore_index=True).to_csv(ANA/'cluster_covariate_composition.tsv.gz',sep='\t',index=False)
    pd.DataFrame(cluster_qc).to_csv(ANA/'cluster_qc.tsv',sep='\t',index=False)
    marker=sparse.load_npz(ANA/'marker_log1p_normalized.npz')
    marker_info=json.loads((ANA/'marker_genes.json').read_text())
    assert marker_info['cell_order_sha256']==digest_lines(obj.obs_names.tolist())
    detection=marker.copy(); detection.data=np.ones_like(detection.data)
    marker_frames=[]
    for key in ['SubCellType']+keys:
        labels=pd.Categorical(obj.obs[key]); groups=labels.categories.astype(str)
        membership=sparse.csr_matrix((np.ones(obj.n_obs,dtype=np.float32),(labels.codes,np.arange(obj.n_obs))),shape=(len(groups),obj.n_obs))
        number=np.bincount(labels.codes,minlength=len(groups))
        means=(membership@marker).toarray()/number[:,None]
        fractions=(membership@detection).toarray()/number[:,None]
        for i,group in enumerate(groups):
            for j,gene in enumerate(marker_info['genes']):
                marker_frames.append({'grouping':key,'group':group,'n_cells':int(number[i]),'gene':gene,'mean_log1p_normalized_expression':float(means[i,j]),'fraction_expression_gt_zero':float(fractions[i,j])})
    pd.DataFrame(marker_frames).to_csv(ANA/'marker_expression_mean_fraction.tsv',sep='\t',index=False)
    component_count,component=sparse.csgraph.connected_components(obj.obsp['connectivities'],directed=False)
    component_size=np.bincount(component)
    obj.obs['graph_component']=component.astype(str)
    pd.DataFrame({'component':np.arange(component_count),'n_cells':component_size}).to_csv(ANA/'graph_components.tsv',sep='\t',index=False)
    qc={'n_cells':obj.n_obs,'n_hvgs':obj.n_vars,'all_umap_finite':bool(np.isfinite(obj.obsm['X_umap']).all()),'coordinate_set_sha256':digest_lines(sorted(obj.obs_names.tolist())),'graph_connected_components':int(component_count),'largest_component_n_cells':int(component_size.max()),'singleton_components':int((component_size==1).sum()),'cells_removed':0,'cluster_counts':{k:int(obj.obs[k].nunique()) for k in keys},'umap_range':{'min':obj.obsm['X_umap'].min(axis=0).tolist(),'max':obj.obsm['X_umap'].max(axis=0).tolist()},'interpretation':'Author subtype annotations are preserved independently of new Leiden clusters. Marker summaries are descriptive only. UMAP spacing and island positions are not quantitative biological distances. Sample-specific clusters require interpretation against composition and QC; Harmony does not establish absence of batch effects.'}
    write_json(ANA/'embedding_qc.json',qc)
    save_checkpoint(obj,'embedding.h5ad')

def analyze():
    sc.settings.n_jobs=16; sc.settings.verbosity=2
    parameters={'seed':SEED,'threads':16,'cell_selection':'Exactly 369063 cell_name records in existing scTCR-covered T-cell CSV; no additional outlier removal or subsampling','input_counts':'Raw GSE236581_counts.mtx.gz, aligned by exact cell_name barcode','hvg':{'n_top_genes':3000,'flavor':'seurat_v3','batch_key':'Ident','span':0.3,'minimum_cells_expressing_gene':3,'excluded_gene_regex':'^TR[ABGD][VDJ]','exclusion_rationale':'Avoid receptor V/D/J usage directly driving expression geometry; these genes remain available for marker summaries'},'normalization':{'target_sum':10000,'transform':'log1p'},'scale':{'zero_center':True,'max_value':10},'pca':{'n_comps':50,'svd_solver':'randomized','geometry_recomputed_on_selected_cells':True},'harmony':{'batch_key':'Ident','n_pcs':50,'theta':2,'lamb':1,'nclust':100,'max_iter_harmony':20,'max_iter_kmeans':10},'neighbors':{'n_neighbors':30,'n_pcs':30,'representation':'X_pca_harmony','metric':'euclidean','transformer':'pynndescent'},'leiden':{'resolutions':[0.4,0.8,1.2],'flavor':'igraph','n_iterations':2,'directed':False,'default_display_resolution':0.8,'labels_are_descriptive':True},'umap':{'min_dist':0.3,'spread':1.0,'n_components':2,'maxiter':300,'init_pos':'spectral','negative_sample_rate':5,'uses_recomputed_graph':True},'rationale':'Subset-specific HVGs/PCA/neighbors replace the full-atlas axes. Thirty PCs and 30 neighbors are a fixed smoothing/denoising choice, checked by explained variance and cluster/subtype/sample composition. No parameters are selected to force subtype agreement or hide scattered points. All observed source cells and all 32 source subtype annotations remain.','software':{p:version(p) for p in ['scanpy','anndata','numpy','scipy','harmonypy','umap-learn','pynndescent','igraph','leidenalg','scikit-learn','scikit-misc']},'python':platform.python_version(),'marker_statistics':'Mean log1p normalized expression and fraction >0; no post-clustering marker p-values are used as inference'}
    if (ANA/'hvg_adaptation.json').exists():
        recovered=json.loads((ANA/'hvg_adaptation.json').read_text())
        if 'effective_flavor' in recovered:
            parameters['hvg'].update({'flavor':recovered['effective_flavor'],'batch_key':recovered['effective_batch_key'],'span':recovered['effective_span'],'n_learning_cells':recovered['n_learning_cells'],'final_strategy':recovered['final_strategy'],'adaptation_file':'hvg_adaptation.json','effective_grouping_file':'hvg_effective_batch_grouping.tsv'})
            parameters['hvg'].update({'effective_batch_count':recovered.get('effective_batch_count'),'n_cells_excluded_from_feature_learning_only':recovered.get('n_cells_excluded_from_hvg_learning_only')})
    write_json(ANA/'parameters.json',parameters)
    if (ANA/'embedding.h5ad').exists() and (ANA/'coordinates.tsv.gz').exists():
        log('Embedding and coordinates already complete'); return
    with threadpool_limits(limits=16):
        if (ANA/'graph_checkpoint.h5ad').exists():
            obj=sc.read_h5ad(ANA/'graph_checkpoint.h5ad')
        else:
            obj=correct_and_graph(prepare_pca())
        cached_umap=ANA/'umap_coordinates.npy'
        if cached_umap.exists():
            obj.obsm['X_umap']=np.load(cached_umap)
        else:
            log('Computing fresh spectral-initialized UMAP for all cells, 300 epochs, min_dist=0.3')
            sc.tl.umap(obj,min_dist=0.3,spread=1.0,n_components=2,maxiter=300,init_pos='spectral',random_state=SEED)
            np.save(cached_umap,obj.obsm['X_umap'])
        summarize(obj)
    log('ANALYSIS COMPLETE')

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--extract-only',action='store_true'); parser.add_argument('--wait-for-counts',action='store_true'); args=parser.parse_args()
    ANA.mkdir(parents=True,exist_ok=True)
    if args.wait_for_counts:
        log('Waiting for independent raw count extraction checkpoint')
        deadline=time.time()+21600
        while not (ANA/'selected_counts.h5ad').exists():
            if time.time()>deadline: raise TimeoutError('Counts checkpoint not ready within 6 h')
            time.sleep(10)
    else:
        extract()
    if args.extract_only: return
    analyze()

if __name__=='__main__':
    main()
