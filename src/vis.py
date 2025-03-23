import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from adjustText import adjust_text


def pmid2gse(df: pd.DataFrame):
    df_sorted = df.sort_values("Cluster_Name")

    df_sorted["PMID"] = df_sorted["PMID"].astype(str)
    df_sorted["GSE_ID"] = df_sorted["GSE_ID"].astype(str)

    plt.figure(figsize=(25, 25))

    sns.scatterplot(
        x="PMID",
        y="GSE_ID",
        hue="Cluster_Name",
        data=df_sorted,
        palette="hsv",
        s=150,
        edgecolor="black",
        alpha=0.8,
    )

    plt.grid(True, linestyle="--", linewidth=0.6)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("PMID", fontsize=10, labelpad=100)
    plt.ylabel("GSE_ID", fontsize=10, labelpad=100)
    plt.title("PMID and GSE_ID with GSE_ID Clusters", fontsize=18)

    plt.legend(
        title="Cluster Name", bbox_to_anchor=(1.15, 1), loc="upper left"
    )  # Further from the plot
    plt.tight_layout(pad=20.0)  # Even more padding to avoid label clipping

    return plt


def pca_vis(df):
    fig, ax = plt.subplots(figsize=(15, 10))

    sns.scatterplot(
        x="PCA1",
        y="PCA2",
        hue="Cluster_Name",
        data=df,
        palette="hsv",
        s=150,
        edgecolor="black",
        alpha=0.8,
        ax=ax,
    )

    texts = []
    for i, row in df.iterrows():
        texts.append(
            ax.text(row["PCA1"], row["PCA2"], row["GSE_ID"], fontsize=8, color="black")
        )

    adjust_text(
        texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5), force_points=0.5
    )

    ax.set_title("PCA Visualisation of GSE_ID Clusters", fontsize=14)
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(title="Cluster Name", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()

    return fig
