BDD <- read.csv("/Users/rezisabashvili/Desktop/BDDD.csv", sep = ";", fileEncoding = "latin1")

head(BDD)

View(BDD)

noms_colonnes <- colnames(BDD)

noms_colonnes_distincts <- unique(gsub("\\.actionDetails\\.\\d+", "", noms_colonnes))

print(noms_colonnes_distincts)
