suppressPackageStartupMessages({library(data.table);library(GSVA);library(BiocParallel);library(survival);library(ggplot2)})
options(stringsAsFactors=FALSE,warn=1)
setDTthreads(2)
root <- Sys.getenv('CRC_TCGA_ROOT')
stopifnot(nzchar(root), dir.exists(root))
sigpath <- Sys.getenv('CRC_SIGNATURE_GMT')
stopifnot(nzchar(sigpath), file.exists(sigpath))
stopifnot(system2('sha256sum',shQuote(sigpath),stdout=TRUE)[1] == paste('32992a8abdefd4a990e1d3d470e2d18e2707bb8d9d34e0456da97c6622097465',sigpath,sep='  '))
lines <- strsplit(readLines(sigpath),'\t'); gsets <- setNames(lapply(lines,function(x)unique(x[-c(1,2)])),vapply(lines,`[`,character(1),1))
stopifnot(length(gsets)==4,all(lengths(gsets)==30))
for (d in c('scores','plots','model_diagnostics')) dir.create(file.path(root,d),showWarnings=FALSE)
writeLines(c(capture.output(sessionInfo()),'ssgseaParam formals',capture.output(formals(ssgseaParam))),file.path(root,'R_session_info.txt'))
inv<-fread(file.path(root,'cohort_inventory.tsv'));sm<-fread(file.path(root,'sample_selection_manifest.tsv'));cl<-fread(file.path(root,'clinical_patient_manifest.tsv'),na.strings=c('','NA'));cdr<-fread(file.path(root,'TCGA-CDR.tsv'),na.strings=c('','NA','#N/A','[Not Available]','[Not Applicable]'))
stopifnot(!anyDuplicated(paste(cl$project,cl$patient)))
stopifnot(!anyDuplicated(cdr$bcr_patient_barcode))
for (nm in c('OS','OS.time','PFI','PFI.time')) cdr[,(nm):=suppressWarnings(as.numeric(get(nm)))]
cdr<-cdr[,.(patient=bcr_patient_barcode,CDR_type=type,CDR_OS_event=OS,CDR_OS_time_days=OS.time,PFI_event=PFI,PFI_time_days=PFI.time,CDR_Redaction=Redaction,CDR_stage=ajcc_pathologic_tumor_stage)]
read_expr<-function(project){
 p<-inv$expression_path[inv$project==project]
 d<-fread(p,check.names=FALSE,showProgress=FALSE);gn<-as.character(d[[1]])
 bc<-sm$barcode[sm$project==project & sm$selected==TRUE]
 stopifnot(!anyDuplicated(bc),all(bc %in% names(d)))
 if(!length(bc)) return(NULL)
 x<-as.matrix(d[,..bc]);storage.mode(x)<-'double';rownames(x)<-gn
 stopifnot(all(is.finite(x)),all(x>=0))
 if(anyDuplicated(gn)){tab<-table(gn);x<-rowsum(x,group=gn,reorder=FALSE);x<-x/as.numeric(tab[rownames(x)])}
 x[order(rownames(x)),,drop=FALSE]
}
cohorts<-c('CRC',sub('TCGA-','',inv$project[!inv$project %in% c('TCGA-COAD','TCGA-READ')]))
coverage<-list();scoresall<-list();cohortlogs<-list();result<-list();warnlog<-list()
addwarn<-function(txt,cohort,sig,endpoint,model){warnlog[[length(warnlog)+1L]]<<-data.table(cohort=cohort,signature=sig,endpoint=endpoint,model=model,warning=txt)}
runfit<-function(expr,cohort,sig,endpoint,model){tryCatch(withCallingHandlers(expr,warning=function(w){addwarn(conditionMessage(w),cohort,sig,endpoint,model);invokeRestart('muffleWarning')}),error=function(e){addwarn(conditionMessage(e),cohort,sig,endpoint,model);NULL})}
for (cohort in cohorts){
 cat(format(Sys.time()),'START',cohort,'\n');flush.console()
 projects<-if(cohort=='CRC') c('TCGA-COAD','TCGA-READ') else paste0('TCGA-',cohort)
 bc<-sm[project %in% projects & selected==TRUE,.(project,patient,barcode)]
 if(!nrow(bc)){cohortlogs[[length(cohortlogs)+1L]]<-data.table(cohort=cohort,status='no_primary_01_samples',n=0);next}
 cache<-file.path(root,'scores',paste0(cohort,'_ssgsea_scores.tsv'))
 if(file.exists(cache)){
  scored<-fread(cache);stopifnot(all(unique(scored$signature)%in%names(gsets)))
 } else {
  xx<-lapply(projects,read_expr);shared<-Reduce(intersect,lapply(xx,rownames));x<-do.call(cbind,lapply(xx,function(m)m[shared,,drop=FALSE]));rm(xx);gc()
  stopifnot(!anyDuplicated(substr(colnames(x),1,12)),!anyDuplicated(colnames(x)),all(bc$barcode %in% colnames(x)))
  isconst<-matrixStats::rowMins(x)==matrixStats::rowMaxs(x)
  genes_before<-nrow(x);x<-log2(x[!isconst,,drop=FALSE]+1)
  cov<-rbindlist(lapply(names(gsets),function(s)data.table(cohort=cohort,signature=s,frozen_genes=length(gsets[[s]]),genes_present=sum(gsets[[s]]%in%rownames(x)),coverage_fraction=mean(gsets[[s]]%in%rownames(x)),missing_genes=paste(setdiff(gsets[[s]],rownames(x)),collapse=';'),genes_in_matrix=genes_before,genes_used=nrow(x),constant_genes_removed=sum(isconst))))
  fwrite(cov,file.path(root,'scores',paste0(cohort,'_gene_coverage.tsv')),sep='\t');coverage[[length(coverage)+1L]]<-cov
  valid<-cov[genes_present>=5 & coverage_fraction>=0.5,signature]
  stopifnot(length(valid)==4)
  param<-ssgseaParam(x,gsets[valid],alpha=.25,normalize=TRUE,minSize=5,checkNA='yes',use='all.obs')
  es<-gsva(param,verbose=FALSE,BPPARAM=MulticoreParam(workers=2,progressbar=FALSE))
  scored<-rbindlist(lapply(rownames(es),function(s){v<-as.numeric(es[s,]);cut<-median(v);data.table(cohort=cohort,signature=s,barcode=colnames(es),patient=substr(colnames(es),1,12),score=v,cutpoint=cut,group=ifelse(v>cut,'High','Low'),equals_median=v==cut)}))
  fwrite(scored,cache,sep='\t');rm(x,param,es);gc()
 }
 scored<-merge(scored,bc[,.(project,patient,barcode)],by=c('patient','barcode'),all.x=TRUE,sort=FALSE)
 dd<-merge(scored,cl,by=c('project','patient'),all.x=TRUE,sort=FALSE)
 dd<-merge(dd,cdr,by='patient',all.x=TRUE,sort=FALSE)
 dd[,clinical_match:=!is.na(indexed_file)|!is.na(XML_patient_file)]
 dd[,CDR_eligible:=!is.na(CDR_type) & (is.na(CDR_Redaction)|tolower(CDR_Redaction)!='redacted') & paste0('TCGA-',CDR_type)==project]
 for(nm in c('OS_time_days','OS_event','stage','age'))dd[,(nm):=suppressWarnings(as.numeric(get(nm)))]
 dd[,high:=as.integer(group=='High')]
 scoresall[[length(scoresall)+1L]]<-dd
 cohortlogs[[length(cohortlogs)+1L]]<-data.table(cohort=cohort,status='scored',n=nrow(bc),signature_count=uniqueN(scored$signature),clinical_patients=uniqueN(dd[clinical_match==TRUE,patient]),CDR_patients=uniqueN(dd[CDR_eligible==TRUE,patient]))
 for(sig in names(gsets)){
  ds<-dd[signature==sig]; med<-unique(ds$cutpoint);stopifnot(length(med)==1)
  for(endpoint in c('Stage','OS_GDC','OS_CDR_sensitivity','PFI_CDR')){
   rr<-data.table(cohort=cohort,family=ifelse(cohort=='CRC','CRC_primary','other_cancers_exploratory'),signature=sig,state=sub('CRC19_(.*)_TOP30','\\1',sig),signature_role=ifelse(grepl('_c08_',sig),'secondary','primary'),endpoint=endpoint,expression_n=nrow(ds),median_cutpoint=med,median_ties=sum(ds$equals_median),n=0L,n_low=0L,n_high=0L,events=NA_integer_,events_low=NA_integer_,events_high=NA_integer_,status='unavailable',reason='',effect=NA_real_,ci_lower=NA_real_,ci_upper=NA_real_,p=NA_real_,cox_wald_p=NA_real_,PH_group_p=NA_real_,PH_global_p=NA_real_,adjusted_n=NA_integer_,adjusted_events=NA_integer_,adjusted_status='not_applicable',adjusted_reason='',adjusted_effect=NA_real_,adjusted_ci_lower=NA_real_,adjusted_ci_upper=NA_real_,adjusted_p=NA_real_,adjusted_PH_group_p=NA_real_,adjusted_PH_global_p=NA_real_,stage_high_early=NA_integer_,stage_high_late=NA_integer_,stage_low_early=NA_integer_,stage_low_late=NA_integer_)
   if(endpoint=='Stage'){
    ds1<-ds[stage %in% 1:4];rr[,`:=`(n=nrow(ds1),n_low=sum(ds1$high==0),n_high=sum(ds1$high==1))]
    if(nrow(ds1)<40||min(rr$n_low,rr$n_high)<15)rr[,reason:='below_fixed_patient_group_threshold']
    else if(uniqueN(ds1$stage>2)<2)rr[,reason:='only_one_stage_category']
    else{
     tab<-matrix(c(sum(ds1$high==1&ds1$stage>2),sum(ds1$high==1&ds1$stage<=2),sum(ds1$high==0&ds1$stage>2),sum(ds1$high==0&ds1$stage<=2)),nrow=2,byrow=TRUE,dimnames=list(c('High','Low'),c('Late','Early')))
     ft<-fisher.test(tab,alternative='two.sided');rr[,`:=`(status='tested',effect=as.numeric(ft$estimate),ci_lower=ft$conf.int[1],ci_upper=ft$conf.int[2],p=ft$p.value,stage_high_late=tab[1,1],stage_high_early=tab[1,2],stage_low_late=tab[2,1],stage_low_early=tab[2,2])]
    }
   }else{
    prefix<-switch(endpoint,OS_GDC='OS',OS_CDR_sensitivity='CDR_OS',PFI_CDR='PFI')
    ds1<-copy(ds);ds1[,`:=`(time=get(paste0(prefix,'_time_days')),event=get(paste0(prefix,'_event')))]
    if(endpoint!='OS_GDC')ds1<-ds1[CDR_eligible==TRUE]
    ds1<-ds1[is.finite(time)&time>0&event %in% c(0,1)]
    rr[,`:=`(n=nrow(ds1),n_low=sum(ds1$high==0),n_high=sum(ds1$high==1),events=sum(ds1$event),events_low=sum(ds1$event[ds1$high==0]),events_high=sum(ds1$event[ds1$high==1]),adjusted_status='unavailable')]
    if(nrow(ds1)<40||min(rr$n_low,rr$n_high)<15||rr$events<10||min(rr$events_low,rr$events_high)<3){rr[,`:=`(reason='below_fixed_patient_group_event_threshold',adjusted_reason='primary_endpoint_ineligible')]}
    else{
     sf<-runfit(survdiff(Surv(time,event)~high,data=ds1),cohort,sig,endpoint,'logrank')
     cf<-runfit(coxph(Surv(time,event)~high,data=ds1,ties='efron',x=TRUE),cohort,sig,endpoint,'univariable')
     if(!is.null(sf)&&!is.null(cf)){
      ss<-summary(cf);ph<-runfit(cox.zph(cf),cohort,sig,endpoint,'univariable_PH');rr[,`:=`(status='tested',effect=ss$conf.int[1,'exp(coef)'],ci_lower=ss$conf.int[1,'lower .95'],ci_upper=ss$conf.int[1,'upper .95'],p=pchisq(sf$chisq,1,lower.tail=FALSE),cox_wald_p=ss$coefficients[1,'Pr(>|z|)'])]
      if(!is.null(ph))rr[,`:=`(PH_group_p=ph$table['high','p'],PH_global_p=ph$table['GLOBAL','p'])]
      saveRDS(list(univariable=cf,PH=ph),file.path(root,'model_diagnostics',paste(cohort,sig,endpoint,'rds',sep='.')))
     }else rr[,reason:='model_error_see_warning_log']
     adj<-ds1[is.finite(age)&sex %in% c('female','male')&stage %in% 1:4];adj[,`:=`(sex=factor(sex),stage_factor=factor(stage))]
     rr[,`:=`(adjusted_n=nrow(adj),adjusted_events=sum(adj$event))]
     if(nrow(adj)<40||min(sum(adj$high==0),sum(adj$high==1))<15)rr[,adjusted_reason:='insufficient_complete_cases']
     else if(uniqueN(adj$sex)<2||uniqueN(adj$stage)<2||uniqueN(adj$age)<2)rr[,adjusted_reason:='prespecified_covariate_missing_or_constant']
     else{
      ncoef<-ncol(model.matrix(~high+age+sex+stage_factor,data=adj))-1
      if(sum(adj$event)<10*ncoef)rr[,adjusted_reason:='less_than_10_events_per_coefficient']
      else{
       af<-runfit(coxph(Surv(time,event)~high+age+sex+stage_factor,data=adj,ties='efron',x=TRUE),cohort,sig,endpoint,'adjusted')
       if(!is.null(af)){
        ass<-summary(af);aph<-runfit(cox.zph(af),cohort,sig,endpoint,'adjusted_PH')
        rr[,`:=`(adjusted_status='tested',adjusted_effect=ass$conf.int['high','exp(coef)'],adjusted_ci_lower=ass$conf.int['high','lower .95'],adjusted_ci_upper=ass$conf.int['high','upper .95'],adjusted_p=ass$coefficients['high','Pr(>|z|)'])]
        if(!is.null(aph))rr[,`:=`(adjusted_PH_group_p=aph$table['high','p'],adjusted_PH_global_p=aph$table['GLOBAL','p'])]
        saveRDS(list(adjusted=af,PH=aph),file.path(root,'model_diagnostics',paste(cohort,sig,endpoint,'adjusted.rds',sep='.')))
       }else rr[,adjusted_reason:='model_error_see_warning_log']
      }
     }
    }
   }
   result[[length(result)+1L]]<-rr
  }
 }
 fwrite(rbindlist(result,fill=TRUE),file.path(root,'results_progress.tsv'),sep='\t')
 cat(format(Sys.time()),'END',cohort,'\n');flush.console()
}
res<-rbindlist(result,fill=TRUE)
res[,q_BH:=p.adjust(p,'BH'),by=.(family,endpoint)]
res[,adjusted_q_BH:=p.adjust(adjusted_p,'BH'),by=.(family,endpoint)]
res[,nominal_adverse:=status=='tested' & is.finite(p) & p<.05 & effect>1]
res[,figure_candidate:=nominal_adverse & endpoint!='OS_CDR_sensitivity']
res[,corrected_adverse:=nominal_adverse & q_BH<.05]
setorder(res,cohort,signature,endpoint)
fwrite(res,file.path(root,'all_results.tsv'),sep='\t')
fwrite(res[figure_candidate==TRUE],file.path(root,'figure_candidates.tsv'),sep='\t')
fwrite(rbindlist(cohortlogs,fill=TRUE),file.path(root,'cohort_status.tsv'),sep='\t')
fwrite(rbindlist(scoresall,fill=TRUE),file.path(root,'patient_scores_clinical_manifest.tsv'),sep='\t')
covs<-rbindlist(lapply(list.files(file.path(root,'scores'),pattern='_gene_coverage.tsv$',full.names=TRUE),fread))
fwrite(covs,file.path(root,'signature_gene_coverage.tsv'),sep='\t')
if(length(warnlog))fwrite(rbindlist(warnlog),file.path(root,'model_warnings.tsv'),sep='\t')
writeLines(c(paste('Completed',format(Sys.time(),tz='UTC')),paste('Tested',sum(res$status=='tested')),paste('Figure candidates',sum(res$figure_candidate))),file.path(root,'ANALYSIS_COMPLETE.txt'))
print(res[cohort=='CRC',.(cohort,state,endpoint,n,events,effect,ci_lower,ci_upper,p,q_BH,adjusted_effect,adjusted_p)])
print(res[figure_candidate==TRUE,.(cohort,state,endpoint,n,events,effect,p,q_BH,adjusted_effect,adjusted_p)])
