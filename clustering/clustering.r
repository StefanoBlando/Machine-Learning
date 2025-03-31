# Load the libraries
library(tclust)
library(cluster)
library(FactoMineR)
library(factoextra)
library(mice)
library(heatmaply)
library(fpc)

# votes.repub is in the workspace 

# Check the structure of the dataset
str(votes.repub)
head(votes.repub)
tail(votes.repub)

# Extract the column names
years <- colnames(votes.repub)
countries <-rownames(votes.repub)

# Select columns from 1900 onwards
votes_from1900 <- votes.repub[, grepl("^X\\d{4}$", years) & as.numeric(gsub("X", "", years)) >= 1900]

#counting NA 
missing_values_count <- rowSums(is.na(votes_from1900))
print(missing_values_count)

#NA cleaning
votes_from1900_clean <- na.omit(votes_from1900)

# years standard deviation and mean
years_sd <- apply(votes_from1900_clean,1,sd)
years_mean <- apply(votes_from1900_clean,1,mean)

#structure
str(votes_from1900_clean)
head(votes_from1900_clean)
tail(votes_from1900_clean)
summary(t(votes_from1900_clean))

#correlation based distance 
res.dist <- get_dist(votes_from1900_clean, method = "pearson")
head(round(as.matrix(res.dist), 2))[, 1:6]

fviz_dist(res.dist, lab_size = 8)

# cluster visualization - ELBOW
wcss <- numeric(length = 10)
for (k in 1:10) {
  kmeans_result <- kmeans(votes_from1900_clean, centers = k, nstart = 25)
  wcss[k] <- kmeans_result$tot.withinss
}

# Plotting the Elbow Method
plot(1:10, wcss, type = "b", pch = 19, frame = FALSE, xlab = "Cluster (k)", ylab = "WCSS")

# cluster visualization - PCA
cor(votes_from1900_clean)

pca_result<- prcomp(votes_from1900_clean, scale=TRUE)
summary(pca_result)

fviz_eig(pca_result)

fviz_pca_ind(pca_result, col.ind = "cos2", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), repel = TRUE)

#PCA GRAPHS
# Years
fviz_pca_var(
  pca_result,
  col.var = "contrib",
  repel = TRUE )

# States
fviz_pca_ind(
  pca_result,
  col.ind = "cos2",
  repel = TRUE)

# Biplot of individuals and variables
fviz_pca_biplot(pca_result, repel = TRUE)

#PAIRS
# Pairs before PCA
pairs(votes_from1900_clean, panel=panel.smooth, col="#6da7a7")

# Pairs after PCA
pairs(pca_result$x, panel=panel.smooth, col="#6da7a7")


# HEATMAP
heatmaply(votes_from1900_clean)


# K-MEANS 
km_result <- kmeans(votes_from1900_clean, centers=2)
fviz_cluster(list(data=votes_from1900_clean, cluster=km_result$cluster))

# cluster centroids 
km_centroids <- km_result$centers

# cluster validation - SILHOUETTE 
fviz_nbclust(votes_from1900_clean, FUN = kmeans)

sil <- silhouette(km_result$cluster, dist(votes_from1900_clean))
fviz_silhouette(sil)

# cluster validation- RESIDUALS
wss <- sapply(1:nrow(votes_from1900_clean), function(i) sum(km_result$centers[km_result$cluster[i], ] - votes_from1900_clean[i, ])^2)

plot(wss, main = "Residuals analysis", xlab = "Index", ylab = "WSS")

#cluster calidation - Dunn Index
clust_stats <- cluster.stats(dist(votes_from1900_clean), km_result$cluster)

print(clust_stats$dunn)

# optimal number of clusters = 2
km_result$nbclust

# CTL 
curves_countries<- ctlcurves(votes_from1900_clean)
plot(curves_countries)
    

# Trimming
tkm_result <- tkmeans(votes_from1900_clean, 2, alpha = 0.1)  

fviz_cluster(list(data = votes_from1900_clean, cluster = tkm_result$cluster))


tkm <- t(tkm_result$centers)

# cluster centroids 
tkm_centroids <- tkm_result$centers



