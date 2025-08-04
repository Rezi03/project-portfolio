# Mémoire - Learning Analytics 
# Analyses préliminaires de la base de données pour le 08/01/2025


# Chargement des données 

## Chargement des packages nécessaires pour analyser les données 
library(shiny)            # Nécessaire pour la librairie Factoshiny
library(ggplot2)          # Nécessaire pour la librairie Factoshiny
library(igraph)           # Nécessaire pour la librairie DataExplorer
library(FactoMineR)       # Pour réaliser l'ACP
library(FactoInvestigate) # Interprétation automatique des résultats de l'ACP avec FactoMineR
library(DataExplorer)     # Pour des analyses rapides uni et bi-variées
library(Factoshiny)       # Outil d'exploration interactive des résultats de l'ACP avec FactoMineR
library(dplyr)
library(factoextra)
library(htmltools)
library(corrplot)
library(readr)
library(tidyverse)
library(DataExplorer)
library(tidyr)
library(cluster)
library(ggplot2)

## Chargement du jeu de données utilisé 
BDDA <- read.csv2(file = "~/Desktop/Analyse_BDD/BDDA.csv", fileEncoding = "UTF-8")

### Visualiser le jeu de données 
view(BDDA)


# Structure de la base de données d'origine (aucune modification)
str(BDDA)

## Compter le nombre de colonnes de chaque type
types_counts <- table(sapply(BDDA, class))

## Afficher les résultats
print(types_counts) 

## Résumer les résultats (phrase)
cat("Votre base de données contient :\n")
for (type in names(types_counts)) {
  cat("-", types_counts[type], "colonnes de type", type, "\n")
}
  # Nous avons (base de donnés sans modifications) : 
    ## 2394 colonnes de type character (texte ou des chaînes de caractères)
    ## 1532 colonnes de type integer (nombres entiers)
    ## 192 colonnes de type logical (valeurs booléennes pour exprimer la logique, ici pour NA ?)


# Les visiteurs uniques

## Nombre de visiteurs uniques
unique_visitors <- length(unique(BDDA$fingerprint))
cat("Nombre de visiteurs uniques :", unique_visitors, "\n") # 2340

## Nombre total de lignes dans la colonne fingerprint
total_visits <- length(BDDA$fingerprint)
cat("Nombre total de visites (lignes) :", total_visits, "\n") # 2870

## Vérification des doublons (pour être certain)
duplicate_visits <- total_visits - unique_visitors
cat("Nombre de doublons :", duplicate_visits, "\n") # 530 doublons 


# Les durées (moyenne, min, max) par visiteur unique

## Les colonnes contenant "timeSpent" 
time_columns <- grep("^timeSpent", colnames(BDDA), value = TRUE)
time_columns <- time_columns[!grepl("Pretty", time_columns)]  # Exclure celles contenant "Pretty" - sinon doublons

## Vérification du nombre de colonnes détectées (normalement 218, ce que j'ai compté)
num_time_columns <- length(time_columns)
cat("Nombre de colonnes détectées pour 'timeSpent' sans 'Pretty':", num_time_columns, "\n") # Il y en a bien 218

## Calcul de la durée totale par visite (en secondes)
total_time_per_visit <- rowSums(BDDA[, time_columns], na.rm = TRUE)

## Un dataframe avec 'fingerprint' et la durée totale
time_data <- data.frame(fingerprint = BDDA$fingerprint, total_time = total_time_per_visit)

## Agréger les données par 'fingerprint' pour sommer les durées des doublons
aggregated_time_data <- aggregate(total_time ~ fingerprint, data = time_data, sum)

## Statistiques sur les durées agrégées : moyenne, minimum, maximum
mean_time <- mean(aggregated_time_data$total_time, na.rm = TRUE)
min_time <- min(aggregated_time_data$total_time, na.rm = TRUE)
max_time <- max(aggregated_time_data$total_time, na.rm = TRUE)

## Convertir les secondes en minutes et secondes
convert_to_min_sec <- function(seconds) {
  minutes <- floor(seconds / 60)
  remaining_seconds <- seconds %% 60
  paste0(minutes, " min ", remaining_seconds, " sec")
}

## Les résultats
cat("Durée moyenne des visites :", convert_to_min_sec(mean_time), "\n") # environ 5 min 18 sec
cat("Durée minimale des visites :", convert_to_min_sec(min_time), "\n") # 0
cat("Durée maximale des visites :", convert_to_min_sec(max_time), "\n") # 294 min 27 sec (4 h 54 min 24 sec)


# Les 10 utilisateurs qui ont passé le plus de temps sur le site (combien ?)

## Créer un DataFrame avec 'fingerprint' et la durée totale par visite (déja fait plus haut)
#time_data <- data.frame(fingerprint = BDDA$fingerprint, total_time = total_time_per_visit)

## Agréger les durées des visites pour chaque 'fingerprint' (utilisateur unique)
total_time_per_user <- aggregate(total_time ~ fingerprint, data = time_data, FUN = sum, na.rm = TRUE)

## Renommage des colonnes
colnames(total_time_per_user) <- c("fingerprint", "TotalTime")

## Tri des utilisateurs (décroissant)
top_10_users <- total_time_per_user[order(total_time_per_user$TotalTime, decreasing = TRUE), ]

## Sélection des 10 premiers
top_10_users <- head(top_10_users, 10)

## Convertir les secondes en format minutes et secondes
convert_to_min_sec <- function(seconds) {
  minutes <- floor(seconds / 60)
  remaining_seconds <- seconds %% 60
  paste0(minutes, " min ", remaining_seconds, " sec")
}

## Affichage des 10 utilisateurs (plus de temps sur le site)
cat("Les 10 utilisateurs ayant passé le plus de temps sur le site :\n")
for (i in 1:nrow(top_10_users)) {
  cat("Utilisateur Fingerprint:", top_10_users[i, "fingerprint"], " - Temps total:", convert_to_min_sec(top_10_users[i, "TotalTime"]), "\n")
}


# Nombre d'utilisateurs distincts ayant passé 0 min sur le site

## DataFrame avec 'fingerprint' et la durée totale par visite (déja fait)
#time_data <- data.frame(fingerprint = BDDA$fingerprint, total_time = total_time_per_visit)

### Filtrer les utilisateurs ayant passé 0 seconde (0 min)
zero_time_users <- time_data[time_data$total_time == 0, ]

### Nombre d'utilisateurs
distinct_zero_time_users <- length(unique(zero_time_users$fingerprint))

### Affichage
cat("Nombre d'utilisateurs distincts ayant passé 0 min sur le site :", distinct_zero_time_users, "\n") # 1104 sur 2340 (explique la moyenne basse) - 1234


# Uniquement les individus dont le temps est supérieur à 0 

## Filtrer les utilisateurs ayant passé plus de 0 secondes sur le site
positive_time_users <- time_data[time_data$total_time > 0, ]

## Calcul des statistiques
mean_time_positive <- mean(positive_time_users$total_time, na.rm = TRUE)
min_time_positive <- min(positive_time_users$total_time, na.rm = TRUE)
max_time_positive <- max(positive_time_users$total_time, na.rm = TRUE)

## Conversion des secondes en minutes et secondes
convert_to_min_sec <- function(seconds) {
  minutes <- floor(seconds / 60)
  remaining_seconds <- seconds %% 60
  paste0(minutes, " min ", remaining_seconds, " sec")
}

## Affichage des résultats
cat("Durée moyenne des visites (temps > 0) :", convert_to_min_sec(mean_time_positive), "\n") # environ 7 min 20 sec
cat("Durée minimale des visites (temps > 0) :", convert_to_min_sec(min_time_positive), "\n") # 1 sec 
cat("Durée maximale des visites (temps > 0) :", convert_to_min_sec(max_time_positive), "\n") # 149 min 27 sec


# Nombre d'individus ayant moins de 2 minutes sur le site

## Filtrer les utilisateurs
less_than_2_minutes <- time_data[time_data$total_time < 2 * 60, ]

## Compter le nombre d'individus 
num_less_than_2_minutes <- nrow(less_than_2_minutes)

## Affichage 
cat("Nombre d'individus ayant passé moins de 2 minutes sur le site :", num_less_than_2_minutes, "\n") # 1990


# Uniquement les individus dont le temps est supérieur à 2 min 

## Filtrer les utilisateurs 
more_than_2_minutes <- time_data[time_data$total_time > 2 * 60, ]

## Calcul 
mean_time_over_2 <- mean(more_than_2_minutes$total_time, na.rm = TRUE)
min_time_over_2 <- min(more_than_2_minutes$total_time, na.rm = TRUE)
max_time_over_2 <- max(more_than_2_minutes$total_time, na.rm = TRUE)

## Convertir les secondes en format minutes et secondes
convert_to_min_sec <- function(seconds) {
  minutes <- floor(seconds / 60)
  remaining_seconds <- seconds %% 60
  paste0(minutes, " min ", remaining_seconds, " sec")
}

## Affichage des résultats
cat("Durée moyenne pour les utilisateurs ayant passé plus de 2 minutes :", convert_to_min_sec(mean_time_over_2), "\n") # environ 13 min 23 sec
cat("Durée minimale pour les utilisateurs ayant passé plus de 2 minutes :", convert_to_min_sec(min_time_over_2), "\n") # 2 min 1 sec
cat("Durée maximale pour les utilisateurs ayant passé plus de 2 minutes :", convert_to_min_sec(max_time_over_2), "\n") # Idem


# Nombre de pages vues

## Sélectionner toutes les colonnes dont le nom commence par 'pageIdAction (actionDetails'
pageIdAction_columns <- grep("^pageIdAction", colnames(BDDA), value = TRUE)

## Afficher le nombre de colonnes sélectionnées
cat("Nombre de colonnes commençant par 'pageIdAction", length(pageIdAction_columns), "\n") # Il y en a 218

## "pageIdAction_total" qui fait la somme de toutes les colonnes "pageIdAction"
BDDA <- BDDA %>%
  mutate(pageIdAction_total = rowSums(select(., all_of(pageIdAction_columns)), na.rm = TRUE))  # Somme des colonnes 'pageIdAction'

## Les 10 visites avec le plus grand total de pages vues
top_10_visits <- BDDA %>%
  group_by(idVisit) %>%  # Regrouper par 'idVisit'
  summarise(pageIdAction_total = max(pageIdAction_total, na.rm = TRUE)) %>%  # Obtenir le total par visite
  arrange(desc(pageIdAction_total)) %>%  # Trier par total décroissant
  head(10)  # Sélectionner les 10 premières lignes

## Afficher les résultats
cat("Les 10 premières visites avec le plus grand nombre total de pages vues :\n")
print(top_10_visits)

## Extraire les "fingerprint" associés aux 10 premières visites
top_10_fingerprints <- BDDA %>%
  filter(idVisit %in% top_10_visits$idVisit) %>%  # Filtrer les lignes correspondant aux 10 premières visites
  select(fingerprint, idVisit, pageIdAction_total)  # Sélectionner les colonnes 'fingerprint', 'idVisit' et 'pageIdAction_total'

## Afficher
cat("Les fingerprints associés aux 10 premières visites avec le plus grand nombre total de pages vues :\n")
print(top_10_fingerprints)


# Le cheminement

## Sélectionner uniquement les colonnes qui commencent par "pageTitle"
pageTitle_columns <- BDDA %>%
  select(starts_with("pageTitle"))

## Fusionner les colonnes 
pageTitle_combined <- pageTitle_columns %>%
  unite("combined_path", everything(), sep = " | ", na.rm = TRUE)

## Nettoyez les valeurs vides 
combined_path <- BDDA %>%
  select(starts_with("pageTitle")) %>%
  unite("combined_path", everything(), sep = " | ", na.rm = TRUE)

## Ignorer les lignes sans chemin (filtre)
combined_path <- combined_path %>%
  filter(combined_path != "")  # Supprime les cheminements vides

## Le cheminement le plus fréquent
most_frequent_path <- combined_path %>%
  group_by(combined_path) %>%
  summarise(n = n()) %>%
  arrange(desc(n))

# Afficher + nombre d'occurrences
print(most_frequent_path)


# Cheminement identique par visiteur ? 

## Créer un cheminement pour chaque individu en ajoutant 'fingerprint'
combined_path_individual <- BDDA %>%
  select(fingerprint, starts_with("pageTitle")) %>%
  unite("combined_path", starts_with("pageTitle"), sep = " | ", na.rm = TRUE) 

## Filtrer les doublons pour chaque individu, vérifier si le même cheminement est répété
path_repetition <- combined_path_individual %>%
  group_by(fingerprint, combined_path) %>%
  summarise(count = n()) %>%
  filter(count > 1) %>%
  arrange(desc(count))

## Afficher les cheminements répétés par individu
print(path_repetition)


# Temps par action

## Sélectionner toutes les colonnes "timeSpent" et calculer la moyenne pour chacune
timeSpent_columns <- BDDA %>%
  select(starts_with("timeSpent"))

## Calculer la moyenne pour chaque colonne
timeSpent_means <- timeSpent_columns %>%
  summarise(across(everything(), ~ mean(. , na.rm = TRUE)))

## Afficher les 10 premières moyennes
top_10_timeSpent_means <- timeSpent_means %>%
  pivot_longer(everything(), names_to = "column", values_to = "mean") %>%
  arrange(desc(mean)) %>%
  head(10)

## Afficher les 10 premières moyennes
print(top_10_timeSpent_means)


# Utilisation d'un clustering 

## Sélectionner les colonnes "timeSpent"
timeSpent_columns <- grep("^timeSpent", colnames(BDDA), value = TRUE)
timeSpent_columns <- timeSpent_columns[!grepl("Pretty", timeSpent_columns)]  # Exclure celles contenant "Pretty"

## Extraire les données de "timeSpent"
time_data <- BDDA %>%
  select(all_of(timeSpent_columns))

## Remplacer les NA par des zéros
time_data[is.na(time_data)] <- 0

## Appliquer un clustering
set.seed(123)  # Pour la reproductibilité
kmeans_result <- kmeans(time_data, centers = 3)

## Ajouter les résultats du clustering
BDDA$cluster <- kmeans_result$cluster

## Visualisation des clusters
ggplot(BDDA, aes(x = time_data[,1], y = time_data[,2], color = factor(cluster))) +
  geom_point() +
  labs(title = "Clustering K-means des utilisateurs",
       x = "TimeSpent Action1", y = "TimeSpent Action2", color = "Cluster")
