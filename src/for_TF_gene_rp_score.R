wd<-"/mnt/"
library(Seurat)
Args <- commandArgs(trailingOnly = TRUE)
# print(Args[1])
TF_file<-read.table(Args[1])

gene_rp_score<-Read10X_h5(Args[2])
tf_rp_score = gene_rp_score[TF_file$V1,]
write.table(tf_rp_score,paste(wd, "TF_rp_score.txt",sep=''),sep="\t",quote=FALSE,row.names=FALSE)
print("finish for_TF_gene_rp_score.R")