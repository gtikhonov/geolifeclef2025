library(Hmsc)
library(jsonify)
library(pROC)
library(abind)
path_data = Sys.getenv("GLC_DATA_PATH")
if (path_data == "") stop("GLC_DATA_PATH environment variable not set")

RS = 1
nChains = 1
fmDir = "fmTF_mahti"

ns = 2519
np = 100
nf = 20
nSamples = 100
thin = 4000
batchSize = 1000
covNum = 228            # 228=90%, 408=95%

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


# load posterior --------------------------------------------------------------------------------------------------------------------------------
transient = nSamples * thin
fitTF_file_name = sprintf("TF_%s_chain01_sam%.4d_thin%.4d.rds", modelTypeString, nSamples, thin)
print(fitTF_file_name)
importFromHPC = from_json(readRDS(file = file.path(path_data, "hmsc", fmDir, fitTF_file_name))[[1]])
postList = importFromHPC[1:nChains]
fitTF = importPosteriorFromHPC(m, postList, nSamples, thin, transient)
BetaPost = getPostEstimate(fitTF, "Beta")
if(np > 0){
  alphaPost = getPostEstimate(fitTF, "Alpha")
  print(alphaPost)
}


if(covNum == "orig"){
  cat("read test covariates\n")
  covTest = read.csv(file.path(path_data, "hmsc", "test_cov.csv"))
} else{
  cat("read test deep features\n")
  covTest = read.csv(file.path(path_data, "hmsc", "test_deepfeatures.csv"))
}
lonlatTest = covTest[, c("lon", "lat")]
XDataTest = covTest[, setdiff(colnames(covTest), c("lon", "lat"))]
if(covNum == "orig" || covNum == "deep"){
  XDataTest = XDataTest
} else{
  XDataTest = as.data.frame(predict(pc, XDataTest))
}

predMean = matrix(NA, nrow=nrow(XDataTest), ncol=ncol(Y))
startTime = proc.time()
pb = txtProgressBar(max=ceiling(nrow(XDataTest)/batchSize), style=3)
setTxtProgressBar(pb, 0)
for(b in 1:ceiling(nrow(XDataTest)/batchSize)){
  start = (b-1)*batchSize + 1
  end = min(b*batchSize, nrow(XDataTest))
  XDataBatch = XDataTest[start:end,]
  if(np > 0){
    EPS = 1e-6
    lonlatBatch = lonlatTest[start:end,] + EPS*matrix(rnorm((end-start+1)*2), nrow=end-start+1, ncol=2)
    xyExt = rbind(xy, lonlatBatch)
    rownames(xyExt) = sprintf("%.4d", (1:nrow(xyExt))-1)
    studyDesignExt = data.frame(val=as.factor(rownames(xyExt)[-(1:nrow(xy))]))
    colnames(studyDesignExt) = sprintf("kmeans%d", np)
    rLExt = HmscRandomLevel(sData=xyExt)
    rLExtList = list(rLExt)
    names(rLExtList) = sprintf("kmeans%d", np)
  } else{
    rLExtList = list(); 
    studyDesignExt = NULL
  }
  pred = predict(fitTF, XData=XDataBatch, studyDesign=studyDesignExt, ranLevels=rLExtList, expected=TRUE, predictEtaMean=TRUE)
  predMean[start:end,] = colMeans(abind(pred, along=0))
  setTxtProgressBar(pb, b)
}
close(pb)
print(proc.time() - startTime)

predBase = matrix(colMeans(Yall), nrow=nrow(XDataTest), ncol=ncol(Yall), byrow=TRUE)
indCommon = order(colSums(Yall), decreasing=TRUE)[1:m$ns]
predBase[,indCommon] = predMean
dir.create(file.path(path_data, "hmsc", "pred"), showWarnings=FALSE, recursive=TRUE)
predFileName = sprintf("pred_%s_sam%.4d_thin%.4d.csv", modelTypeString, nSamples, thin)
predBase = round(predBase, 3)
write.csv(predBase, file.path(path_data, "hmsc", "pred", predFileName), row.names=FALSE)


b = 1
start = (b-1)*batchSize + 1
end = min(b*batchSize, nrow(XData))
XDataBatch = XData[start:end,]
if(np > 0){
  EPS = 1e-6
  lonlatBatch = lonlatTest[start:end,] + EPS*matrix(rnorm((end-start+1)*2), nrow=end-start+1, ncol=2)
  xyExt = rbind(xy, lonlatBatch)
  rownames(xyExt) = sprintf("%.4d", (1:nrow(xyExt))-1)
  studyDesignExt = data.frame(val=as.factor(rownames(xyExt)[-(1:nrow(xy))]))
  colnames(studyDesignExt) = sprintf("kmeans%d", np)
  rLExt = HmscRandomLevel(sData=xyExt)
  rLExtList = list(rLExt)
  names(rLExtList) = sprintf("kmeans%d", np)
} else{
  rLExtList = list()
  studyDesignExt = NULL
}
pred = predict(fitTF, XData=XDataBatch, studyDesign=studyDesignExt, ranLevels=rLExtList, expected=TRUE, predictEtaMean=TRUE)
predBatch = colMeans(abind(pred, along=0))
aucVec = rep(NA, m$ns)
for(j in 1:m$ns){
  if(length(unique(Y[start:end,j])) > 1){
    aucVec[j] = auc(roc(Y[start:end,j], predBatch[,j], direction="<", quiet=TRUE))
  }
}
plot(aucVec, main=modelTypeString, xlab="species", ylab="AUC", ylim=c(0.5,1))


