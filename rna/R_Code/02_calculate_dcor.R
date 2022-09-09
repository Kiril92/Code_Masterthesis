# Setup -------------------------------------------------------------------
library(stringr)
library(dplyr)
library(tidyr)
library(Rfast)

# Userinputs --------------------------------------------------------------

# filename_rawdata <- "data/agg_gene_data_short_f100.csv"
filename_rawdata <- "data/agg_gene_data_short_ALL.csv"

# n_correlation_rows <- 200

# Lese benoetigte Daten ein -----------------------------------------------

geneData <- read.csv(file = filename_rawdata)

n_spalten <- ncol(geneData) 
n_correlation_rows <- n_spalten

# Paarweise Berechnung der Distanzkorrelationen ---------------------------

# Allokiere Korrelationsmatrix
corr_matrix <- matrix(0,
                    nrow = n_correlation_rows,
                    ncol = n_correlation_rows)

t1 <- Sys.time()
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
t2 <- Sys.time();t2-t1
# NA-Einträge zu 0
corr_matrix[is.nan(corr_matrix)] <- 0

# Setze Diagoneinträge auf 1
diag(corr_matrix) <- 1

# Sicherung der Korrelationsmatrix ----------------------------------------

# Dateiname im Format "corr_matrix_1000_2021-11-17.csv"
filename_corrmatrix <- paste0("data/correlation_matrices/corr_matrix_",
                              n_correlation_rows, "_",
                              Sys.Date(), ".csv")

# Sichere berechnete Matrix in .csv-Datei
message("Sichere Korrelationsmatrix unter: ", filename_corrmatrix)
write.table(corr_matrix,
            file = filename_corrmatrix,
            row.names = FALSE,
            col.names = FALSE)
