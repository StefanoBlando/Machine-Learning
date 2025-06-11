# Voting Patterns Clustering Analysis

This project applies advanced clustering techniques to analyze historical republican voting patterns across different countries. By implementing various dimensionality reduction and clustering algorithms, the analysis identifies underlying patterns and similarities in voting behavior over time.

## Project Overview

Voting behavior analysis provides valuable insights into international relations and political alignments. This project explores a dataset of republican votes from 1900 onwards, using clustering techniques to identify countries with similar voting patterns and how these patterns have evolved over time.

## Dataset

The analysis uses the `votes.repub` dataset which contains:
- Countries as rows
- Years (from 1900 onwards) as columns
- Voting percentages as values

## Methodologies

The project implements and compares the following techniques:

### Exploratory Data Analysis
- Missing value analysis and handling
- Descriptive statistics (mean, standard deviation)
- Correlation-based distance calculation

### Dimensionality Reduction
- Principal Component Analysis (PCA)
- Visualization of principal components
- Contribution and quality of representation analysis

### Clustering Techniques
- K-means clustering
- Trimmed K-means (robust clustering)
- Cluster Trimmed Likelihood curves (CTL)

### Cluster Validation
- Elbow method for optimal cluster number
- Silhouette analysis
- Dunn index
- Residuals analysis

## Key Findings

- Two distinct clusters of countries emerged based on voting patterns
- PCA revealed significant temporal patterns in the data
- Trimmed K-means provided more robust clustering by handling outliers effectively

## Repository Structure

```
voting-patterns-analysis/
├── setup.R                   # Setup file with package installation
├── src/                   
│   ├── data_preparation.R    # Data cleaning and preparation
│   ├── exploratory_analysis.R # Exploratory data analysis
│   ├── pca_analysis.R        # PCA implementation
│   ├── kmeans_clustering.R   # K-means clustering
│   └── robust_clustering.R   # Trimmed K-means and CTL analysis
└── README.md                # Project documentation
```

## Prerequisites

This project requires R with the following packages:
- tclust
- cluster
- FactoMineR
- factoextra
- mice
- heatmaply
- fpc

## Usage

To run the analysis:

```r
# First, run the setup file to install packages and load data
source("setup.R")

# Then run individual analysis files as needed
source("src/data_preparation.R")
source("src/exploratory_analysis.R")
source("src/pca_analysis.R")
source("src/kmeans_clustering.R")
source("src/robust_clustering.R")
```

## Visualizations

The project includes several key visualizations:
- Correlation heatmap of voting patterns
- PCA biplots showing countries and years
- Cluster visualization with dimensionality reduction
- Silhouette plots for cluster validation
- Cluster Trimmed Likelihood curves

## Future Work

Potential improvements include:
- Incorporating additional political and economic variables
- Time series analysis to track cluster evolution
- Comparative analysis with alternative clustering techniques
- Geospatial visualization of voting pattern clusters

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The data providers for the republican voting dataset
- R package developers for the clustering and analysis tools
