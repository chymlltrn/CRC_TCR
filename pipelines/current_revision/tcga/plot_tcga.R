suppressPackageStartupMessages({library(data.table);library(survival)})
root <- Sys.getenv('CRC_TCGA_ROOT')
stopifnot(nzchar(root), dir.exists(root))
formats <- strsplit(Sys.getenv('CRC_TCGA_PLOT_FORMATS', 'pdf,png,svg'), ',', fixed=TRUE)[[1]]
stopifnot(length(formats)>0L, all(formats %in% c('pdf','png','svg')))
if(any(formats %in% c('png','svg'))) {
 # A compiled-in Cairo capability can still have missing runtime libraries.
 probe <- tempfile(fileext='.png'); previous_device <- dev.cur()
 cairo_ok <- tryCatch(withCallingHandlers({
   png(probe, type='cairo', width=100, height=100); grid::grid.newpage(); dev.off()
   file.exists(probe) && file.info(probe)$size > 0
 }, warning=function(w) stop(conditionMessage(w))), error=function(e) FALSE)
 if(dev.cur()!=previous_device) dev.off()
 unlink(probe)
 if(!isTRUE(cairo_ok)) stop('PNG/SVG require working R Cairo libraries; use CRC_TCGA_PLOT_FORMATS=pdf for PDF only')
}
r<-fread(file.path(root,'all_results.tsv'));d<-fread(file.path(root,'patient_scores_clinical_manifest.tsv'))
r[,state_short:=sub('^(c[0-9]+).*','\\1',state)]
cols<-c(c27='#32948E',c03='#3B6FB6',c20='#DB8651',c08='#9A6FAD')
rankstates<-c('c27','c03','c20','c08')
pfmt<-function(x){ifelse(is.na(x),'NA',ifelse(x<.001,formatC(x,format='e',digits=1),formatC(x,format='f',digits=3)))}
valid_cox<-r$endpoint=='Stage' | (is.finite(r$effect)&is.finite(r$ci_lower)&is.finite(r$ci_upper)&r$ci_lower>0&r$ci_upper>0)
r[,plot_eligible:=figure_candidate==TRUE & valid_cox]
fwrite(r[figure_candidate==TRUE],file.path(root,'figure_candidates_plot_checked.tsv'),sep='\t')
open_device<-function(path,ext,w,h){switch(ext,pdf=pdf(paste0(path,'.pdf'),width=w,height=h,family='Helvetica',useDingbats=FALSE),png=png(paste0(path,'.png'),width=w,height=h,units='in',res=180,type='cairo'),svg=svg(paste0(path,'.svg'),width=w,height=h,family='sans'))}
for(ep in c('Stage','OS_GDC','PFI_CDR')){
 z<-r[endpoint==ep&plot_eligible==TRUE]
 if(!nrow(z))next
 z[,state_order:=match(state_short,rankstates)];setorder(z,state_order,cohort)
 n<-nrow(z);title<-switch(ep,Stage='Later pathologic stage',OS_GDC='Worse overall survival',PFI_CDR='Worse progression-free interval')
 method<-if(ep=='Stage')'Stage III/IV vs I/II; Fisher exact test' else 'Log-rank test; Cox hazard ratio'
 measure<-if(ep=='Stage')'Odds ratio (High vs Low)' else 'Hazard ratio (High vs Low)'
 h<-max(4.8,3.05+n*.29);path<-file.path(root,'plots',paste0('TCGA_',ep,'_all_adverse_candidates_forest'))
 low<-max(.08,min(z$ci_lower[is.finite(z$ci_lower)]));high<-min(50,max(z$ci_upper[is.finite(z$ci_upper)]))
 low<-min(.8,low);high<-max(2,high)
 axisvals<-c(.1,.2,.5,1,2,5,10,20,50);axisvals<-axisvals[axisvals>=low&axisvals<=high]
 map<-function(v)27+28*(log(pmax(low,pmin(high,v)))-log(low))/(log(high)-log(low))
 for(ext in formats){
  open_device(path,ext,11,h);par(mar=c(5.2,.5,3.2,.5),family='sans',xpd=NA)
  plot.new();plot.window(xlim=c(0,113),ylim=c(-1.7,n+1.4));title(main=paste0('TCGA | ',title),line=1.5,adj=0,cex.main=1.25)
  mtext(paste0('Nominal P < 0.05 and effect > 1; fixed ssGSEA signatures and within-cohort median split'),side=3,line=.2,adj=0,cex=.78,col='#555555')
  abline(h=seq(.5,n+.5,by=2),col='#F3F3F3',lwd=16)
  segments(map(1),.5,map(1),n+.5,col='#A0A0A0',lty=2)
  text(c(0,56.5,74,82,92,103),n+1,c('Cancer / state','Effect (95% CI)','n','Events','P','q / q_all'),adj=c(0,.5),font=2,cex=.78)
  for(i in seq_len(n)){
   y<-n-i+1;color<-cols[z$state_short[i]]
   text(0,y,paste0(paste(z$cohort[i],z$state_short[i],sep=' / '),ifelse(!is.na(z$PH_group_p[i])&&z$PH_group_p[i]<.05,'*','')),adj=0,cex=.83,col=color,font=2)
   segments(map(z$ci_lower[i]),y,map(z$ci_upper[i]),y,col=color,lwd=1.4)
   points(map(z$effect[i]),y,pch=15,cex=.85,col=color)
   if(!is.finite(z$ci_upper[i])||z$ci_upper[i]>high)arrows(map(high)-.6,y,map(high),y,length=.05,col=color)
   text(56.5,y,sprintf('%.2f (%.2f-%.2f)',z$effect[i],z$ci_lower[i],z$ci_upper[i]),adj=0,cex=.72)
   text(75.5,y,z$n[i],cex=.76);text(84,y,if(ep=='Stage')'-' else z$events[i],cex=.76)
   text(94,y,pfmt(z$p[i]),cex=.73)
   text(105,y,paste(pfmt(z$q_family_BH[i]),pfmt(z$q_all_main_secondary_BH[i]),sep=' / '),cex=.70)
  }
  axis(1,at=map(axisvals),labels=axisvals,pos=.25,cex.axis=.72,tck=-.02,mgp=c(3,.45,0))
  text(41,-1.2,measure,cex=.79)
  mtext(paste0(method,'. q: BH across all four signatures and non-CRC cohorts for this endpoint.'),side=1,line=2.1,adj=0,cex=.66)
  mtext('q_all: BH across all Stage, OS and PFI tests; CDR OS sensitivity excluded. Complete results remain in audit.',side=1,line=3.1,adj=0,cex=.66)
  if(any(z$PH_group_p<.05,na.rm=TRUE))mtext('* Proportional-hazards test P < 0.05: HR is a time-averaged summary; association P is from the log-rank test.',side=1,line=4.1,adj=0,cex=.66)
  dev.off()
 }
}
# Examples are deterministic: CRC first, then c27/c03/c20/c08, then cancer alphabetic; max two survival examples.
examples<-r[plot_eligible==TRUE & endpoint %in% c('OS_GDC','PFI_CDR')]
examples[,`:=`(crc_order=as.integer(cohort!='CRC'),state_order=match(state_short,rankstates),ep_order=match(endpoint,c('OS_GDC','PFI_CDR')))];setorder(examples,crc_order,state_order,cohort,ep_order)
# One example per cancer-state avoids using both related endpoints as apparently independent evidence.
examples<-examples[!duplicated(paste(cohort,signature))][seq_len(min(2,.N))]
for(ii in seq_len(nrow(examples))){
 a<-examples[ii];pref<-if(a$endpoint=='OS_GDC')'OS' else 'PFI'
 z<-d[cohort==a$cohort&signature==a$signature];z[,`:=`(time=get(paste0(pref,'_time_days')),event=get(paste0(pref,'_event')))]
 if(a$endpoint=='PFI_CDR')z<-z[CDR_eligible==TRUE]
 z<-z[is.finite(time)&time>0&event %in% c(0,1)];z[,group:=factor(group,levels=c('Low','High'))]
 fit<-survfit(Surv(time,event)~group,data=z);xmax<-min(max(z$time),365.25*10);at<-pretty(c(0,xmax/365.25),n=5);at<-at[at>=0&at<=xmax/365.25]
 risk<-summary(fit,times=at*365.25,extend=TRUE)
 path<-file.path(root,'plots',paste0('TCGA_example_',a$cohort,'_',a$state_short,'_',a$endpoint,'_KM'))
 for(ext in formats){
  open_device(path,ext,6.2,6.5);layout(matrix(1:2,ncol=1),heights=c(4.5,1.5));par(mar=c(4,4.3,3.5,1),family='sans')
  plot(fit,col=c('#307FA9','#BF4E4E'),lwd=2,conf.int=FALSE,mark.time=TRUE,xlim=c(0,xmax),xscale=365.25,xlab='Time from diagnosis (years)',ylab=if(a$endpoint=='OS_GDC')'Overall survival probability' else 'Progression-free interval probability',cex.axis=.85,cex.lab=.9)
  title(main=paste(a$cohort,'|',gsub('_',' ',a$state)),cex.main=1.1,line=2.1)
  mtext(paste0('High > median; Low <= median | n = ',nrow(z),'; events = ',sum(z$event)),side=3,line=.65,cex=.8)
  legend('topright',legend=paste0(c('Low','High'),' (n=',c(sum(z$group=='Low'),sum(z$group=='High')),')'),col=c('#307FA9','#BF4E4E'),lwd=2,bty='n',cex=.8)
  text(.03*xmax,.12,labels=paste0('HR ',sprintf('%.2f',a$effect),' (95% CI ',sprintf('%.2f',a$ci_lower),'-',sprintf('%.2f',a$ci_upper),')\nLog-rank P = ',pfmt(a$p),'; q = ',pfmt(a$q_family_BH),'; q_all = ',pfmt(a$q_all_main_secondary_BH)),adj=0,cex=.75)
  par(mar=c(2.2,4.3,.2,1));plot.new();plot.window(xlim=c(0,xmax),ylim=c(0,2.7));text(0,2.5,'Number at risk',adj=0,cex=.8)
  for(j in 1:2){ix<-seq.int((j-1)*length(at)+1,j*length(at));text(at*365.25,2-j/1.1,risk$n.risk[ix],cex=.78,col=c('#307FA9','#BF4E4E')[j]);mtext(c('Low','High')[j],side=2,at=2-j/1.1,las=1,line=.6,cex=.8)}
  axis(1,at=at*365.25,labels=at,cex.axis=.8,tick=FALSE);dev.off()
 }
}
fwrite(examples,file.path(root,'deterministic_survival_examples.tsv'),sep='\t')
# One stage illustration by the same deterministic ordering.
z<-r[plot_eligible==TRUE&endpoint=='Stage'];z[,`:=`(crc_order=as.integer(cohort!='CRC'),state_order=match(state_short,rankstates))];setorder(z,crc_order,state_order,cohort)
if(nrow(z)){
 a<-z[1];tab<-matrix(c(a$stage_low_early,a$stage_low_late,a$stage_high_early,a$stage_high_late),nrow=2,dimnames=list(c('I/II','III/IV'),c('Low','High')))
 path<-file.path(root,'plots',paste0('TCGA_example_',a$cohort,'_',a$state_short,'_stage'))
 for(ext in formats){
  open_device(path,ext,5,4.8);par(mar=c(4,4.5,4.5,1));bp<-barplot(prop.table(tab,2),col=c('#AFC8D8','#BF6470'),ylim=c(0,1),ylab='Fraction of patients',xlab='ssGSEA signature group',border=NA)
  text(bp,rep(.05,2),paste0('n = ',colSums(tab)),cex=.85)
  legend('topleft',legend=c('Stage I/II','Stage III/IV'),fill=c('#AFC8D8','#BF6470'),bty='n',horiz=TRUE,cex=.8,inset=c(0,-.2),xpd=NA)
  title(main=paste(a$cohort,'|',gsub('_',' ',a$state)),line=3,cex.main=1.05)
  mtext(paste0('OR ',sprintf('%.2f',a$effect),'; Fisher P = ',pfmt(a$p),'; q = ',pfmt(a$q_family_BH),'; q_all = ',pfmt(a$q_all_main_secondary_BH)),side=3,line=1.3,cex=.66)
  dev.off()
 }
 fwrite(a,file.path(root,'deterministic_stage_example.tsv'),sep='\t')
}
writeLines('All plots use the frozen ssGSEA results. Candidate filtering is nominal P<0.05 and adverse direction, with family and all-scan q-values displayed. Cox figures require finite HR and CI. Review model_warnings.tsv and PH tests before interpretation.',file.path(root,'plots/PLOT_METHODS.txt'))
