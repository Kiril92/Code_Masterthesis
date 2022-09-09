library(dplyr)
library(stringr)

agg_data <- read.csv(file = "data/agg_gene_data_short_f100.csv")
sample_metadata <- read.csv(file = "data/metadaten/samples_metadata/samples_metaDataIR3.csv")

sample_metadata <- do.call(rbind, lapply(split(sample_metadata, sample_metadata$PATNO), head, 1))

sample_metadata <- sample_metadata %>% 
  group_by(PATNO) %>% 
  summarise_all(.funs = first())

# Extract PATNO from filename-column
agg_data <- agg_data %>% 
  mutate(PATNO = str_extract(id, "(?<=IR3\\.)\\d+")) %>% 
  left_join(sample_metadata %>% select(PATNO, DIAGNOSIS),
            by = "PATNO")


