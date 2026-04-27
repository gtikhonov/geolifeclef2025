library(Hmsc)
library(jsonify)
path_data = Sys.getenv("GLC_DATA_PATH")
if (path_data == "") stop("GLC_DATA_PATH environment variable not set")

RS = 1
nChains = 1
nSamples = 10
thin = 1
transient = nSamples * thin

ns = 2269
np = 100
nf = 20
covNum = "orig"            # 228=90%, 408=95%


# read data and preprocess ----------------------------------------------------------------------------------------------------------------------
set.seed(RS)
if(covNum == "orig"){
  cat("read covariates\n")
  cov_pa = read.csv(file.path(path_data, "hmsc", "train_cov.csv"))
  lonlat_pa = cov_pa[, c("lon", "lat")]
  XData_pa = cov_pa[, setdiff(colnames(cov_pa), c("lon", "lat"))]
  cov_po = read.csv(file.path(path_data, "hmsc", "po_X.csv"))
  lonlat_po = cov_po[, c("lon", "lat")]
  XData_po = cov_po[, setdiff(colnames(cov_po), c("cell_id", "class", "lon", "lat"))]
  XData_pa = cbind(data.frame(obs=0, XData_pa))
  XData = rbind(XData_pa, XData_po)
  XData = cbind(data.frame(type=as.factor(c(rep("pa",nrow(XData_pa)), rep("po",nrow(XData_po))))), XData)
  XData$obs = log(XData$obs + 1) / 5
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
if(!exists("Y_pa")){
  cat("read train_Y\n")
  Y_pa = read.csv(file.path(path_data, "hmsc", "train_Y.csv"))
} else{
  cat("reuse train_Y\n")
}
if(!exists("Y_po")){
  cat("read train_Y\n")
  Y_po = read.csv(file.path(path_data, "hmsc", "po_Y.csv"))
} else{
  cat("reuse po_Y\n")
}
indCommon = order(colSums(Y_pa), decreasing=TRUE)[1:ns]
Y = as.matrix(rbind(Y_pa, Y_po))[,indCommon]

train_clusters = read.csv(file.path(path_data, "hmsc", "centroids_po_pa", "train_clusters.csv"))
po_clusters = read.csv(file.path(path_data, "hmsc", "centroids_po_pa", "po_clusters.csv"))
studyDesign = rbind(train_clusters, po_clusters)
colnames(studyDesign) = c("kmeans100", "kmeans200", "kmeans400")
for(i in 1:ncol(studyDesign)) {
  studyDesign[,i] = as.factor(sprintf("%.4d", studyDesign[,i]))
}

# setup Hmsc ------------------------------------------------------------------------------------------------------------------------------------
if(np > 0){
  xy = read.csv(file.path(path_data, "hmsc", "centroids_po_pa", sprintf("centroids_k%d.csv", np)))
  rownames(xy) = sprintf("%.4d", (1:nrow(xy))-1)
  xy = xy[, c("lon", "lat")]
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
  plot(lonlat_pa[,1], lonlat_pa[,2], pch=NA, xlab="lon", ylab="lat")
  text(xy[,1], xy[,2], rownames(xy), col="red")
  text(lonlat_pa[indDisplay,1], lonlat_pa[indDisplay,2], studyDesign[indDisplay, sprintf("kmeans%d", np)])
}
