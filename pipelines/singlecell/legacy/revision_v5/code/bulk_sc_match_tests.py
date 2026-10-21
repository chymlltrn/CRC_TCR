#!/usr/bin/env python3
"""Bounded regression tests for identity, translation, thresholds and read partitions."""
from release_config import release_path, private_value
import unittest,tempfile,json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd,numpy as np
import bulk_sc_match_ingest as m
ROOT=Path(str(release_path('bulk_sc_match_tests_source_8')))
class MatchTests(unittest.TestCase):
 def test_translation_no_truncation(self):
  nt=pd.Series(['TGTGCTTCTTCTTTT','TGCGCGAGTAGCTTC','TGTGCTTCTTCTGTT','TGTGCTTCTTCTTT','TGTGCTTCTTCTTTTN','TGTGCTTCTTCTTTT'])
  aa=pd.Series(['CASSF','CASSF','CASSF','CASSF','CASSF','CASSFL'])
  self.assertEqual(m.nt_compatible(nt,aa).tolist(),[True,True,False,False,False,False])
 def test_genes_no_alias_or_multicall(self):
  v=m.gene(pd.Series([' trbv5-5*01 ','TRBV5-5*01,TRBV5-4*01','TRBV24/OR9-2','BV5-5']))
  j=m.gene(pd.Series(['TRBJ1-4*01']*4))
  self.assertEqual(m.gene_valid(v,j).tolist(),[True,False,True,False]);self.assertEqual(v.iloc[3],'BV5-5')
 def test_tied_gene_vs_allele(self):
  self.assertTrue(m.tied('TRBV5-5*01(100),TRBV5-4*01(100)'))
  self.assertFalse(m.tied('TRBV5-5*01(100),TRBV5-5*02(100)'))
  self.assertFalse(m.tied('TRBV5-5*01(100),TRBV5-4*01(99)'))
 def test_patient_key_threshold_and_partitions(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'bulk_sc_match') as td:
   b=Path(td);o=b/'out';d=b/'data';o.mkdir();d.mkdir()
   for z in ['library_receipts','library_coverage','library_gene_tokens']:(o/z).mkdir()
   (d/'library_support').mkdir();old=m.BULK;m.BULK=b
   aa='CASSF';n1='TGTGCTTCTTCTTTT';n2='TGCGCGAGTAGCTTC';v='TRBV5-5';j='TRBJ1-4'
   vals=[(aa,n1,1,v,True),(aa,n2,1,v,True),(aa,n1,4,'TRBV5-5,TRBV5-4',True),(aa,'TGTGCTTCTTCTGTT',5,v,True),('CATTY','TGTGCTACTACTTAT',3,'TRBV6-1',True),('X',n1,12,v,False)]
   rows=[]
   for i,(a,n,c,vv,q) in enumerate(vals):rows.append(dict(cohort='home',sample_id='TEST',patient_key='old_frozen_patient',clone_id=i,aa=a,nt_cdr3=n,read_count=c,qc_pass=q,v_gene=vv,j_gene=j,all_v_hits_with_score='TRBV5-5*01(100),TRBV5-4*01(100)' if i==0 else vv+'*01(100)',all_j_hits_with_score=j+'*01(100)'))
   pd.DataFrame(rows).to_parquet(b/'home__TEST.parquet',index=False)
   r=SimpleNamespace(cohort='home',sample_id='TEST',patient_key='master_patient',library_qc_reads=14)
   allsets={'AA':{aa},'AA_VJ':{aa+'|'+v+'|'+j},'NT_VJ':{n1+'|'+v+'|'+j,n2+'|'+v+'|'+j}};ts=allsets;ns={x:set() for x in m.DEFS}
   try:rr=m.scan_library(r,allsets,ts,ns,o,d,2)
   finally:m.BULK=old
   self.assertEqual(rr['raw_clone_rows'],6);self.assertEqual(rr['qc_clone_rows'],5);self.assertEqual(rr['qc_reads'],14);self.assertTrue(rr['scan_complete_to_EOF'])
   cov=pd.read_csv(o/'library_coverage/home__TEST.tsv',sep='\t').set_index(['definition','read_threshold'])
   self.assertEqual(cov.loc[('AA_VJ',2),'full_source_matched_reads'],7)
   self.assertEqual(cov.loc[('AA',2),'full_source_matched_reads'],11)
   self.assertEqual(cov.loc[('NT_VJ',2),'full_source_matched_reads'],0)
   self.assertEqual(cov.loc[('NT_VJ',1),'full_source_matched_reads'],2)
   self.assertEqual(cov.loc[('AA_VJ',2),'Tumor_only_matched_reads'],7)
   self.assertEqual(cov.loc[('AA_VJ',2),'no_baseline_reads'],7)
   self.assertEqual(rr['rank_cross_gene_tie_rows'],1)
   s=pd.read_parquet(d/'library_support/home__TEST.parquet');self.assertTrue(s.patient_key.eq('master_patient').all());self.assertEqual(s[(s.definition=='AA_VJ')].rank_tied_rows.sum(),1)
if __name__=='__main__':unittest.main(verbosity=2)
