library(tidyr)
family = read.csv("/Volumes/qtran/Semisupervised_Learning/python/output/all_seeds_RF+NN_balanced_family_summary.csv", header=TRUE)
subclass = read.csv("/Volumes/qtran/Semisupervised_Learning/python/output/all_seeds_RF+NN_balanced_subclass_summary.csv", header=TRUE)

avg_family = family %>% group_by(Dataset, Metric, Validation, Model) %>% summarise(avg = mean(Value), n = n())
avg_subclass = subclass %>%
                group_by (Dataset, Metric, Validation, Model) %>%
                summarise(avg=mean(Value), n = n())
write.csv(avg_family, file = "/Volumes/qtran/Semisupervised_Learning/python/output/average_bacc_RF+NN_family.csv")
write.csv(avg_subclass, file = "/Volumes/qtran/Semisupervised_Learning/python/output/average_bacc_RF+NN_subclass.csv")
