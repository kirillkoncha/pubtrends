# PubTrends: Data Insights for Enhanced Paper Relevance

Service that takes file with a list of PMIDs. Parse descriptions of datasets connected to PMIDs and their IDs (GSE_ID).
Performs clusterisations based on TF-IDF embeddings and visualise results via:
1. Scatter Plot with PMIDs on X-axis and GSE_IDs on Y-axis. Dots indicate connections between PMID and GSE_ID. Dot color indicates cluster of the dataset
2. PCA of clusters based on datasets descriptions

# Running
1. Enter the root directory of the repository
2. Create venv with: `python3 -m venv venv`
3. Entrer the venv with: `source venv/bin/activate`
4. Install requirements with: `python3 -m pip install -r requirements.txt`
5. Run the service with: `streamlit run app.py --server.port 8501`
6. Open `localhost:8501` in your browser
