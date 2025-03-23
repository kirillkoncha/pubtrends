import streamlit as st

from src.clusterisations import clustering_web
from src.get_data import process_pmids
from src.vis import pca_vis, pmid2gse

SAVE_DATASET_PATH = "./data/pmid_gse_data.csv"

st.title("PubTrends: Extraction of Articles and Linked Datasets")

st.session_state["file_uploaded"] = False
st.session_state["parsing_done"] = False
st.session_state["file_lines"] = None
st.session_state["min_clusters"] = None
st.session_state["max_clusters"] = None
st.session_state["clustering_done"] = False

st.subheader("Upload File")
uploaded_file = st.file_uploader(
    "Upload text file with articles PMIDS (each PMID should be on a new line)",
    type=["txt"],
)

if uploaded_file:
    st.write("✅ File uploaded successfully!")
    st.session_state["file_lines"] = uploaded_file.read().decode("utf-8").splitlines()
    st.session_state["file_uploaded"] = True

if st.session_state["file_uploaded"] and not st.session_state["parsing_done"]:
    st.subheader("Clusterisation Parameters")
    st.write("🤔 Choose minimum and maximum number of clusters")

    min_clusters = int(
        st.text_input("Enter the minimum number of clusters (min: 2):", value="2")
    )
    max_clusters = int(
        st.text_input("Enter the maximum number of clusters (min: 3):", value="15")
    )

    st.session_state["min_clusters"] = min_clusters
    st.session_state["max_clusters"] = max_clusters

    if min_clusters < 2:
        st.warning("Minimum number of clusters should be at least 2.")
    elif max_clusters < 3:
        st.warning("Maximum number of clusters should be at least 3.")
    elif max_clusters <= min_clusters:
        st.warning("Maximum number of clusters should be greater than the minimum.")
    else:
        st.write(f"Selected Clustering Range: {min_clusters} - {max_clusters}")

        if st.button("Start Clusterisation"):
            st.write("Parsing Data...")

            progress_bar = st.progress(0)
            progress_placeholder = st.empty()

            process_pmids(
                st.session_state["file_lines"],
                SAVE_DATASET_PATH,
                progress_bar,
                progress_placeholder,
            )
            progress_bar.progress(1.0)
            progress_placeholder.write("✅ Articles parsed successfully!")

            st.session_state["parsing_done"] = True

            df, optimal_k, silhouette_score = clustering_web(
                SAVE_DATASET_PATH, min_clusters, max_clusters
            )
            st.info(
                f"Optimal number of clusters is {optimal_k} with silhouette score equal to {silhouette_score}"
            )

            st.session_state["clustering_done"] = True

if st.session_state["clustering_done"]:
    if "plot_data" not in st.session_state:
        st.session_state["plot_data"] = df

    st.pyplot(pmid2gse(st.session_state["plot_data"]))
    st.pyplot(pca_vis(st.session_state["plot_data"]))

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Clustered Data (Plots Data Will Disappear)",
        data=csv,
        file_name="data_clustered.csv",
        mime="text/csv",
    )
