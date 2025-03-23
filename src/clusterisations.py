import argparse

import numpy as np
import pandas as pd
from scipy.sparse._csr import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances


def vectorize_data(
    df: pd.DataFrame,
    stop_words: str | None = "english",
    max_features: int = 5000,
    vectorizer: TfidfVectorizer | None = None,
) -> tuple[csr_matrix, TfidfVectorizer]:
    """Perform TF-IDF vectorization on combined text."""
    if vectorizer is None:
        vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(df["Combined_Text"])
    else:
        tfidf_matrix = vectorizer.transform(df["Combined_Text"])
    return tfidf_matrix, vectorizer


def find_optimal_k(
    tfidf_matrix: csr_matrix, min_clusters: int, max_clusters: int
) -> tuple[int, float]:
    """Find the optimal number of clusters using Silhouette Score."""
    silhouette_scores = []
    k_values = range(min_clusters, max_clusters)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(tfidf_matrix)
        score = silhouette_score(tfidf_matrix, kmeans.labels_)
        silhouette_scores.append(score)

    optimal_k = k_values[np.argmax(silhouette_scores)]
    return optimal_k, np.max(silhouette_scores)


def cluster_data(
    df: pd.DataFrame, tfidf_matrix: csr_matrix, optimal_k: int
) -> tuple[pd.DataFrame, KMeans]:
    """Perform K-Means clustering, assign cluster labels, and compute distances to centroids."""
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(tfidf_matrix)

    # Compute the centroids of the clusters
    centroids = kmeans.cluster_centers_

    # Calculate the cosine distance between each dataset and its cluster's centroid
    distances_to_centroid = []
    for i in range(len(df)):
        cluster_id = df.iloc[i]["Cluster"]
        dataset_vector = (
            tfidf_matrix[i].toarray().flatten()
        )  # Convert sparse matrix to dense
        centroid_vector = centroids[
            cluster_id
        ]  # Get the centroid vector for the cluster
        distance = cosine_distances([dataset_vector], [centroid_vector])[0][
            0
        ]  # Calculate cosine distance
        distances_to_centroid.append(distance)

    # Add the distance to centroid column
    df["Distance_Centroid"] = distances_to_centroid

    return df, kmeans


def assign_cluster_names(
    df: pd.DataFrame, vectorizer: TfidfVectorizer, kmeans: KMeans, optimal_k: int
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Assign human-readable names to each cluster based on top TF-IDF terms."""
    cluster_names = {}
    terms = np.array(vectorizer.get_feature_names_out())

    for i in range(optimal_k):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        cluster_text = " ".join(df.iloc[cluster_indices]["Combined_Text"])
        tfidf_scores = vectorizer.transform([cluster_text]).toarray()[0]
        top_words = terms[np.argsort(tfidf_scores)[-3:]]  # Top 3 words
        cluster_names[i] = " ".join(top_words)  # Assign name based on top words

    df["Cluster_Name"] = df["Cluster"].map(cluster_names)
    return df, cluster_names


def perform_pca(tfidf_matrix: csr_matrix) -> np.ndarray:
    """Perform PCA for visualization and reduce dimensions to 2D."""
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(tfidf_matrix.toarray())
    return reduced_data


def clustering_web(
    file_path: str, min_clusters: int, max_clusters: int
) -> tuple[pd.DataFrame, int, float]:
    df = pd.read_csv(file_path)
    df = df.drop_duplicates(subset=["GSE_ID"]).reset_index(drop=True)
    df["Combined_Text"] = (
        df[["Title", "Experiment Type", "Summary", "Organism", "Overall Design"]]
        .fillna("")
        .agg(" ".join, axis=1)
    )

    tfidf_matrix, vectorizer = vectorize_data(df)

    optimal_k, silhouette_score = find_optimal_k(
        tfidf_matrix, min_clusters, max_clusters
    )

    df, kmeans = cluster_data(df, tfidf_matrix, optimal_k)

    df, cluster_names = assign_cluster_names(df, vectorizer, kmeans, optimal_k)
    df = df[
        [
            "PMID",
            "GSE_ID",
            "Combined_Text",
            "Cluster",
            "Cluster_Name",
            "Distance_Centroid",
        ]
    ]

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(tfidf_matrix.toarray())
    df["PCA1"] = pca_result[:, 0]
    df["PCA2"] = pca_result[:, 1]

    df.to_csv("clustered_web.csv", index=False)
    return df, optimal_k, silhouette_score


def main():
    # Set up argparse for input/output
    parser = argparse.ArgumentParser(
        description="Clustering and Visualization of Dataset Descriptions"
    )
    parser.add_argument(
        "input_file", type=str, help="Input CSV file with dataset descriptions"
    )
    parser.add_argument(
        "output_file", type=str, help="Output CSV file to save clustering results"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    df = df.drop_duplicates(subset=["GSE_ID"]).reset_index(drop=True)
    df["Combined_Text"] = (
        df[["Title", "Experiment Type", "Summary", "Organism", "Overall Design"]]
        .fillna("")
        .agg(" ".join, axis=1)
    )

    tfidf_matrix, vectorizer = vectorize_data(df)

    optimal_k, silhouette_score = find_optimal_k(tfidf_matrix)
    print(
        f"Optimal number of clusters {optimal_k} with silhouette score equal to {silhouette_score}"
    )

    df, kmeans = cluster_data(df, tfidf_matrix, optimal_k)

    df, cluster_names = assign_cluster_names(df, vectorizer, kmeans, optimal_k)
    df = df[
        [
            "PMID",
            "GSE_ID",
            "Combined_Text",
            "Cluster",
            "Cluster_Name",
            "Distance_Centroid",
        ]
    ]

    reduced_data = perform_pca(tfidf_matrix)
    df["PCA1"] = reduced_data[:, 0]
    df["PCA2"] = reduced_data[:, 1]

    df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
