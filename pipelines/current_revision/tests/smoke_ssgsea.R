suppressPackageStartupMessages({library(GSVA); library(BiocParallel)})
set.seed(110)
x <- matrix(rexp(80*8), nrow=80, dimnames=list(paste0('SYNTH_G',seq_len(80)),paste0('SYNTH_SAMPLE_',seq_len(8))))
sets <- setNames(lapply(seq_len(4),function(i) rownames(x)[seq.int(1+10*(i-1),20+10*(i-1))]),paste0('SYNTH_SET_',seq_len(4)))
p <- ssgseaParam(log2(x+1), sets, alpha=.25, normalize=TRUE, minSize=5, checkNA='yes', use='all.obs')
a <- gsva(p, verbose=FALSE, BPPARAM=SerialParam())
b <- gsva(p, verbose=FALSE, BPPARAM=SerialParam())
stopifnot(identical(dim(a),c(4L,8L)),all(is.finite(a)),identical(a,b))
for(i in seq_len(nrow(a))) {
  cut <- median(a[i,]); group <- ifelse(a[i,]>cut,'High','Low')
  stopifnot(sum(group=='High')==4L,sum(group=='Low')==4L)
}
cat('PASS: deterministic synthetic ssGSEA API and median grouping\n')
