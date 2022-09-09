# Setup -------------------------------------------------------------------

library(dplyr)
library(stringr)

# Userinput ---------------------------------------------------------------

filepath_gene_attributes <- "data/metadaten/gene_attributes/gene_attributes_raw.gtf"

filepath_results <- "data/metadaten/gene_attributes/"

# Lese Daten ein ----------------------------------------------------------

df_gene_attributes <- read.table(file = filepath_gene_attributes,
                                 sep = "\t")

# Extrahiere die relevanten Attribute -------------------------------------

df_gene_attributes$geneId <- str_extract(df_gene_attributes$V9, "ENSG\\d+")
df_gene_attributes$gene_biotype <- str_extract(df_gene_attributes$V9,
                                               "gene_biotype \\w+(_\\w+)?") %>% 
  gsub("gene_biotype ", "", .)

protein_coding_genes_known <- df_gene_attributes[, c("geneId", "gene_biotype")] %>% 
  distinct() %>% 
  filter(gene_biotype == "protein_coding")

# Sichere Tabelle mit `protein-coding` Genen ------------------------------

saveRDS(protein_coding_genes_known, 
     file = paste0(filepath_results, "gene_attributes.rds"))

write.csv(protein_coding_genes_known,
          file = paste0(filepath_results, "gene_attributes.csv"),
          row.names = F)
