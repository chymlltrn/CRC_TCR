suppressPackageStartupMessages({library(data.table);library(survival)})
root <- Sys.getenv('CRC_TCGA_ROOT')
stopifnot(nzchar(root), dir.exists(root))
r<-fread(file.path(root,'all_results.tsv'))
if(!'q_family_BH' %in% names(r))setnames(r,'q_BH','q_family_BH')
r[,q_all_main_secondary_BH:=NA_real_]
r[endpoint!='OS_CDR_sensitivity',q_all_main_secondary_BH:=p.adjust(p,'BH')]
r[,q_all_including_OS_sensitivity_BH:=p.adjust(p,'BH')]
r[,q_BH:=q_family_BH]

if(file.exists(file.path(root,'model_warnings.tsv'))){
 mw<-fread(file.path(root,'model_warnings.tsv'))
 bad<-unique(mw[model=='adjusted' & grepl('infinite|converg',warning,ignore.case=TRUE),.(cohort,signature,endpoint)])
 r[,adjusted_convergence_warning:=paste(cohort,signature,endpoint) %in% paste(bad$cohort,bad$signature,bad$endpoint)]
 r[adjusted_convergence_warning==TRUE,`:=`(adjusted_status='unreliable_convergence_warning',adjusted_reason='Cox covariate coefficient may be infinite; raw estimates retained for audit only')]
 r[,adjusted_p_reliable:=ifelse(adjusted_status=='tested',adjusted_p,NA_real_)]
 r[,adjusted_q_reliable_family_BH:=p.adjust(adjusted_p_reliable,'BH'),by=.(family,endpoint)]
}
fwrite(r,file.path(root,'all_results.tsv'),sep='\t');fwrite(r[figure_candidate==TRUE],file.path(root,'figure_candidates.tsv'),sep='\t')
d<-fread(file.path(root,'patient_scores_clinical_manifest.tsv'));d[,high:=as.integer(group=='High')]
# Additive audit analyses only. Neither model determines candidate figure selection.
audit<-list()
for(co in unique(d$cohort))for(si in unique(d$signature)){
 z<-d[cohort==co&signature==si&stage %in% 1:4]
 if(nrow(z)>=40&&min(sum(z$high==0),sum(z$high==1))>=15&&uniqueN(z$stage)>1){
  w<-wilcox.test(stage~high,data=z,exact=FALSE)
  audit[[length(audit)+1]]<-data.table(cohort=co,signature=si,test='stage_ordered_Wilcoxon_auxiliary',n=nrow(z),events=NA_integer_,effect=median(z[high==1,stage])-median(z[high==0,stage]),ci_lower=NA_real_,ci_upper=NA_real_,p=w$p.value,status='tested',reason='auxiliary_only_not_candidate_selection',PH_group_p=NA_real_,PH_global_p=NA_real_)
 }
 if(co!='CRC')next
 for(ep in c('OS_GDC','OS_CDR_sensitivity','PFI_CDR')){
  pref<-switch(ep,OS_GDC='OS',OS_CDR_sensitivity='CDR_OS',PFI_CDR='PFI')
  z<-d[cohort==co&signature==si];z[,`:=`(time=get(paste0(pref,'_time_days')),event=get(paste0(pref,'_event')))]
  if(ep!='OS_GDC')z<-z[CDR_eligible==TRUE]
  z<-z[is.finite(time)&time>0&event %in% c(0,1)&is.finite(age)&sex %in% c('female','male')&stage %in% 1:4]
  z[,`:=`(sex=factor(sex),stage_factor=factor(stage))]
  a<-data.table(cohort=co,signature=si,test=paste0(ep,'_age_sex_stage_Cox_stratified_COAD_READ'),n=nrow(z),events=sum(z$event),effect=NA_real_,ci_lower=NA_real_,ci_upper=NA_real_,p=NA_real_,status='unavailable',reason='',PH_group_p=NA_real_,PH_global_p=NA_real_)
  if(nrow(z)<40||min(sum(z$high==0),sum(z$high==1))<15||uniqueN(z$sex)<2||uniqueN(z$stage)<2||uniqueN(z$project)<2)a[,reason:='insufficient_complete_cases_or_variation']
  else{
   nc<-ncol(model.matrix(~high+age+sex+stage_factor,data=z))-1
   if(sum(z$event)<10*nc)a[,reason:='less_than_10_events_per_coefficient']
   else{
    fit<-coxph(Surv(time,event)~high+age+sex+stage_factor+strata(project),data=z,x=TRUE,ties='efron');s<-summary(fit);ph<-cox.zph(fit)
    a[,`:=`(status='tested',effect=s$conf.int['high','exp(coef)'],ci_lower=s$conf.int['high','lower .95'],ci_upper=s$conf.int['high','upper .95'],p=s$coefficients['high','Pr(>|z|)'],reason='auxiliary_only_not_candidate_selection',PH_group_p=ph$table['high','p'],PH_global_p=ph$table['GLOBAL','p'])]
    saveRDS(list(fit=fit,PH=ph),file.path(root,'model_diagnostics',paste('CRC',si,ep,'project_stratified.rds',sep='.')))
   }
  }
  audit[[length(audit)+1]]<-a
 }
}
a<-rbindlist(audit,fill=TRUE);a[,q_auxiliary_all_BH:=p.adjust(p,'BH')];fwrite(a,file.path(root,'auxiliary_stage_trend_CRC_project_stratified.tsv'),sep='\t')
writeLines(c('q_family_BH: BH over all tested signatures within CRC or non-CRC family and endpoint.','q_all_main_secondary_BH: BH over all valid Stage, OS_GDC, PFI_CDR tests across all scored cohorts and four signatures.','q_all_including_OS_sensitivity_BH: BH over the same tests plus OS_CDR_sensitivity.','Auxiliary stage ordered-Wilcoxon and CRC project-stratified sensitivity models do not determine figure eligibility; their own complete audit family q is separate.'),file.path(root,'multiplicity_families.txt'))
