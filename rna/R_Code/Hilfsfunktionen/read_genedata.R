#' read_genedata
#' 
#' Ließt eine Featurecounts-Datei und formatiert sie
#' 
#' @param filename string, String mit dem Dateipfad
#' @param sep string, Spalten-Seperator
#' @param header boolean, Beinhaltet die erste Zeile die Spaltennamen
read_genedata <- function(filename, id, sep = "\t", header = T){
  data <- read.table(file = filename,
                     sep = "\t",
                     header = T) %>% 
    dplyr::select(-c(Chr, Strand, Start, End, Length))
  
  colnames(data)[2] <- "reads"
  
  data <- data %>% 
    tidyr::pivot_wider(names_from = Geneid,
                       values_from = reads) %>% 
    mutate(id = id)
  
  return(data)
}