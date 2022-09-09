# Setup -------------------------------------------------------------------
library(stringr)
library(dplyr)
library(tidyr)

source("R_Code/Hilfsfunktionen/read_genedata.R")

# Userinputs --------------------------------------------------------------
filepath_metadata <- "data/metadaten/samples_metadata/samples_metaDataIR3.csv"
filepath_gene_attributes <- "data/metadaten/gene_attributes/gene_attributes.rds"

folderpath_counts <- "data/counts/relevant/"

# Metadaten der Stichprobe ------------------------------------------------
samples_metadata <- read.csv(file = filepath_metadata)

gene_attributes <- readRDS(filepath_gene_attributes)

# Selektiere relevante Featurecount-Dateien -------------------------------
# Zerlege Dateinamen und schreibe Angaben in ein Dataframe
all_files <- list.files(path = folderpath_counts)
all_files <- all_files[grep("featureCounts", all_files)] %>% 
  as.data.frame() %>%
  filter(grepl("featureCounts", .)) %>% 
  setNames("filenames_row") %>% 
  separate(filenames_row,
           c("study", "individual_id", "visit", "sample_id", "seq_facility_id",
             "assay", "analysis", "filetype"),
           "\\.") %>% 
  mutate(individual_id = as.numeric(individual_id),
         filename = list.files(path = folderpath_counts)[grep("featureCounts",
                                                              list.files(path = folderpath_counts))]) %>% 
  left_join(sample_metadata %>% select(PATNO, DIAGNOSIS),
            by = c("individual_id" = "PATNO"))

# Filter Phase1 Daten
relevant_files <- all_files %>% 
  filter(grepl("Phase1", study), # Only Phase 1 data
         visit != "POOL",
         DIAGNOSIS %in% c("PD", "Control")) %>% 
  mutate(individual_id = as.numeric(individual_id),
         sample = str_remove(string = filename, pattern = "\\.featureCounts.*")) %>% 
  distinct(individual_id, .keep_all = TRUE) %>% 
  arrange(DIAGNOSIS, individual_id)

# Übetrage relevante Daten in den `relevant`-Ordner
# for(i in relevant_files$filename){
#   system(paste0("mv data/counts/", i, " data/counts/relevant/", i))
# }

# TODO Filter out "abudantly expressed RN7S cytoplasmic genes" ???
# TODO Tabelle mit Überblick über Gesamtdaten und verwendete Daten

# Lese Gen-Daten ----------------------------------------------------------
# Nach Geneid-matching sind Chr, Start, End, Strand, Length, 

# ls_files <- purrr::map(relevant_files[1:10,], ~print(.))
ls_files <- list()

t1 <- Sys.time()
for(i in 1:nrow(relevant_files)){
  print(i)
  ls_files[[i]] <- read_genedata(filename = paste0(folderpath_counts,relevant_files$filename[i]),
                                 id = relevant_files$sample[i])  
}
t2 <- Sys.time(); t2-t1

geneData <- bind_rows(ls_files)
rm(ls_files)


# Filter: Samples mit counts in >35k Gene ---------------------------------
# from other samples (having read in >35k genes)
genes_counted <- apply(geneData, 1, function(x){sum(x>0)})
df_genes_counted <- data.frame(sample = geneData$id,
                               genes_counted = genes_counted)

# TODO filter sample whose read count distribution were different
geneData_counted <- geneData[df_genes_counted$genes_counted < 35000, ]


# TODO Sichere Tabelle mit |SampleId|GenesCounted|

# Filter: protein-coding Genes --------------------------------------------
# TODO filter protein-coding genes 
clean_gene_names <- gsub("\\..*", "", colnames(geneData))
# Filter protein-coding Gene
geneData_counted_protein <- geneData_counted[, clean_gene_names %in% c(gene_attributes$geneId, "id")]


# DIAGNOSIS hinzufügen ----------------------------------------------------

geneData_counted <- geneData_counted %>% 
  left_join(relevant_files %>% select(sample, DIAGNOSIS),
            by = c("id" = "sample"))

geneData_counted_protein <- geneData_counted_protein %>%
  left_join(relevant_files %>% select(sample, DIAGNOSIS),
            by = c("id" = "sample"))

# Sichere Daten -----------------------------------------------------------
write.csv(geneData,
          # file="data/agg_gene_data_f100.csv",
          file="data/agg_gene_data_ALL.csv",
          row.names = FALSE)

write.csv(geneData_counted_protein,
          # file="data/agg_gene_data_short_f100.csv",
          file="data/agg_gene_data_short_ALL.csv",
          row.names = FALSE
)

write.csv(relevant_files,
          file = "data/agg_gene_metadata_ALL.csv",
          row.names = FALSE)
