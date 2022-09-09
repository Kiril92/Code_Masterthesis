library(foreach)
library(doSNOW)
library(Rfast)


library(doParallel)    
# registerDoParallel(detectCores())
registerDoParallel(3)
getDoParWorkers()


# Userinputs --------------------------------------------------------------

filename_rawdata <- "data/agg_gene_data_short_f100.csv"

n_correlation_rows <- 200

# Lese benoetigte Daten ein -----------------------------------------------

geneData <- read.csv(file = filename_rawdata)

# n_spalten <- ncol(geneData) 
n_spalten <- 100

corr_matrix <- matrix(0, nrow=n_spalten, ncol=n_spalten) 

# Paarweise Berechnung der Distanzkorrelationen ---------------------------

### Parallelisierte Version
# cl <- makeCluster(5, type = "SOCK")
# registerDoSNOW(cl)

t1 <- Sys.time()
result <- foreach (i = 1:n_spalten) %:% # nesting operator
  foreach (j = 1:n_spalten) %dopar% {
    if(j < i){dcor(geneData[,i], geneData[,j])$dcor}
  }
corr_matrix <- matrix(unlist(result), nrow = n_spalten)
t2 <- Sys.time();t2-t1


output <- foreach(k=1:n_spalten) %dopar% {
  for (i in 1:length(t2)) {
    for (j in 1:length(celnum)) {
      corr_matrix[i,j] <- mean(samplefunction(celnum[i],t2[j]))
    }  
  }
  df
}

# Allokiere Korrelationsmatrix
corr_matrix <- matrix(0,
                      nrow = n_correlation_rows,
                      ncol = n_correlation_rows)

for(i in 2:n_correlation_rows){
  
  # Fortschrittsanzeige
  if(i%%10 == 0){
    print(i)
  }
  
  # Berechnung der Distanzkorrelation
  for(j in 1:(i-1)){
    corr_matrix[i,j] <- dcor(geneData[,i], geneData[,j])$dcor
  }
  
}

# NA-Einträge zu 0
corr_matrix[is.nan(corr_matrix)] <- 0

# Setze Diagoneinträge auf 1
diag(corr_matrix) <- 1
