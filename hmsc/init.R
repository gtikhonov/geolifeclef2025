library(Hmsc)
library(jsonify)
path_data = Sys.getenv("GLC_DATA_PATH")
if (path_data == "") stop("GLC_DATA_PATH environment variable not set")

RS = 1
nChains = 1
nSamples = 10
thin = 1
transient = nSamples * thin

use_po = 1
ns = 2519
np = 100
nf = 20
covNum = 408            # 228=90%, 408=95%


# read data and preprocess ----------------------------------------------------------------------------------------------------------------------
set.seed(RS)
if(covNum == "orig"){
  cat("read covariates\n")
  cov = read.csv(file.path(path_data, "hmsc", "train_cov.csv"))
  lonlat = cov[, c("lon", "lat")]
  XData = cov[, setdiff(colnames(cov), c("lon", "lat"))]
} else{
  if(!exists("deepfeatures")){
    cat("read deep features\n")
    deepfeatures = read.csv(file.path(path_data, "hmsc", "train_deepfeatures.csv"))
  } else{
    cat("reuse deep features\n")
  }
  lonlat = deepfeatures[, c("lon", "lat")]
  XData = deepfeatures[, setdiff(colnames(deepfeatures), c("lon", "lat"))]
}
if(covNum == "orig" || covNum == "deep"){
  XData = XData
} else{
  dir.create(file.path(path_data, "hmsc", "pc"), showWarnings=FALSE, recursive=TRUE)
  pc_filename = sprintf("pc_%.4d.RData", covNum)
  pc_filepath = file.path(path_data, "hmsc", "pc", pc_filename)
  if(file.exists(pc_filepath)){
    cat("load PCA\n")
    load(pc_filepath)
  } else{
    cat("compute PCA\n")
    pc = prcomp(XData, rank.=covNum)
    cat("save PCA\n")
    save(pc, file=pc_filepath)
  }
  XData = as.data.frame(pc$x)
}


nc = ncol(XData) + 1
if(np == 0) nf = 0
modelTypeString = sprintf("nc%.4d_ns%.4d_np%.4d_nf%.2d", nc, ns, np, nf)
if(!exists("Yall")){
  cat("read train_Y\n")
  Yall = read.csv(file.path(path_data, "hmsc", "train_Y.csv"))
} else{
  cat("reuse train_Y\n")
}
indCommon = order(colSums(Yall), decreasing=TRUE)[1:ns]
Y = as.matrix(Yall)[,indCommon]

studyDesign = read.csv(file.path(path_data, "hmsc", "kmeans_assignments_grouped.csv"))[,-1]
colnames(studyDesign) = c("kmeans100", "kmeans200", "kmeans400")
for(i in 1:ncol(studyDesign)) {
  studyDesign[,i] = as.factor(sprintf("%.4d", studyDesign[,i]))
}

# setup Hmsc ------------------------------------------------------------------------------------------------------------------------------------
if(np > 0){
  xy = read.csv(file.path(path_data, "hmsc", sprintf("centroids_k%d.csv", np)))
  rownames(xy) = sprintf("%.4d", (1:nrow(xy))-1)
  rL = HmscRandomLevel(sData=xy)
  rL = setPriors(rL, nfMin=nf, nfMax=nf)
  ranLevelsList = list(rL)
  names(ranLevelsList) = sprintf("kmeans%d", np)
} else{
  ranLevelsList = list()
}
cat("initialize Hmsc model\n")
m = Hmsc(Y=Y, XData=XData, XScale=FALSE, distr="probit", studyDesign=studyDesign, ranLevels=ranLevelsList)
# dir.create(file.path(path_data, "hmsc", "unfitted"), showWarnings=FALSE, recursive=TRUE)
# save(m, file=file.path(path_data, "hmsc", "unfitted", sprintf("hm_%s.RData", modelTypeString)))

cat("prepare Hmsc-HPC export\n")
init_obj = sampleMcmc(m, samples=nSamples, thin=thin, transient=transient, nChains=nChains, verbose=1, engine="HPC")
init_obj$hM$Y = NULL
init_obj$hM$XData = NULL
init_obj$hM$X = NULL
init_obj$hM$XScaled = round(init_obj$hM$XScaled, 3)

init_file_name = sprintf("init_%s_chain%.2d.rds", modelTypeString, nChains)
cat(sprintf("%s\n", init_file_name))
dir.create(file.path(path_data, "hmsc", "init"), showWarnings=FALSE, recursive=TRUE)
saveRDS(to_json(init_obj), file=file.path(path_data, "hmsc", "init", init_file_name))
cat("Export to TF saved\n")

if(np > 0){
  indDisplay = 1:10
  plot(lonlat[,1], lonlat[,2], pch=NA, xlab="lon", ylab="lat")
  text(xy[,1], xy[,2], rownames(xy), col="red")
  text(lonlat[indDisplay,1], lonlat[indDisplay,2], studyDesign[indDisplay, sprintf("kmeans%d", np)])
}