# ============================================================
# TASK-02: CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING
# Internship Project
# ============================================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# -----------------------------
# 2. PROJECT PATHS
# -----------------------------

BASE_DIR = Path(__file__).parent

DATA_FILE = BASE_DIR / "Mall_Customers.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create output folder if it does not exist
OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------
# 3. LOAD DATASET
# -----------------------------

print("=" * 60)
print("CUSTOMER SEGMENTATION USING K-MEANS")
print("=" * 60)

print("\n[1] Loading dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully.")


# -----------------------------
# 4. BASIC DATA UNDERSTANDING
# -----------------------------

print("\n[2] Dataset Information")

print(f"Number of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# -----------------------------
# 5. DATA QUALITY CHECK
# -----------------------------

print("\n[3] Data Quality Check")

print("\nMissing values:")
print(df.isnull().sum())

duplicate_count = df.duplicated().sum()

print(f"\nDuplicate rows: {duplicate_count}")


# -----------------------------
# 6. STATISTICAL SUMMARY
# -----------------------------

print("\n[4] Statistical Summary")

print(df.describe())


# -----------------------------
# 7. EXPLORATORY DATA ANALYSIS
# -----------------------------

print("\n[5] Exploratory Data Analysis")

# Annual Income Distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    df["Annual Income (k$)"],
    bins=20,
    kde=True
)

plt.title("Annual Income Distribution")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "income_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Spending Score Distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    df["Spending Score (1-100)"],
    bins=20,
    kde=True
)

plt.title("Spending Score Distribution")
plt.xlabel("Spending Score (1-100)")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "spending_score_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# -----------------------------
# 8. FEATURE SELECTION
# -----------------------------

print("\n[6] Selecting Features")

# We use Annual Income and Spending Score
# because these features are useful for
# identifying customer purchasing behavior.

X = df[
    [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]

print("\nSelected features:")
print(X.head())


# -----------------------------
# 9. VISUALIZE DATA BEFORE CLUSTERING
# -----------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"],
    s=80
)

plt.title("Customers Before Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "customers_before_clustering.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# -----------------------------
# 10. FEATURE SCALING
# -----------------------------

print("\n[7] Scaling Features")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Feature scaling completed.")

print("\nFirst 5 scaled records:")
print(X_scaled[:5])


# -----------------------------
# 11. ELBOW METHOD
# -----------------------------

print("\n[8] Running Elbow Method")

wcss = []

k_values = range(1, 11)

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)


print("\nWCSS values:")

for k, value in zip(k_values, wcss):
    print(f"K = {k}: WCSS = {value:.2f}")


# Elbow graph
plt.figure(figsize=(9, 6))

plt.plot(
    k_values,
    wcss,
    marker="o"
)

plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")

plt.xticks(list(k_values))

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "elbow_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# -----------------------------
# 12. SELECT NUMBER OF CLUSTERS
# -----------------------------

# Based on the traditional elbow analysis
# for this Mall Customers dataset,
# we use 5 clusters.

OPTIMAL_K = 5

print(f"\nSelected number of clusters: K = {OPTIMAL_K}")


# -----------------------------
# 13. SILHOUETTE SCORE
# -----------------------------

print("\n[9] Evaluating Candidate Cluster Counts")

silhouette_scores = {}

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores[k] = score

    print(
        f"K = {k}: "
        f"Silhouette Score = {score:.4f}"
    )


print(
    f"\nSilhouette Score for selected K={OPTIMAL_K}: "
    f"{silhouette_scores[OPTIMAL_K]:.4f}"
)


# -----------------------------
# 14. TRAIN FINAL K-MEANS MODEL
# -----------------------------

print("\n[10] Training Final K-Means Model")

kmeans = KMeans(
    n_clusters=OPTIMAL_K,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(X_scaled)

print("K-Means training completed.")


# -----------------------------
# 15. ADD CLUSTER LABELS
# -----------------------------

df["Cluster"] = labels

print("\nCluster labels assigned successfully.")


# -----------------------------
# 16. CLUSTER COUNTS
# -----------------------------

print("\n[11] Number of Customers in Each Cluster")

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

print(cluster_counts)


# -----------------------------
# 17. GET CLUSTER CENTERS
# -----------------------------

print("\n[12] Calculating Cluster Centers")

# Centers returned by K-Means are scaled
centers_scaled = kmeans.cluster_centers_

# Convert centers back to original units
centers = scaler.inverse_transform(
    centers_scaled
)

cluster_centers = pd.DataFrame(
    centers,
    columns=[
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
)

cluster_centers.index.name = "Cluster"

print("\nCluster Centers:")
print(cluster_centers.round(2))


# Save cluster centers
cluster_centers.to_csv(
    OUTPUT_DIR / "cluster_centers.csv"
)


# -----------------------------
# 18. FINAL CUSTOMER CLUSTER PLOT
# -----------------------------

print("\n[13] Creating Customer Segmentation Plot")

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set1",
    s=100
)

# Plot centroids
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=300,
    c="black",
    label="Centroids"
)

plt.title(
    "Customer Segmentation Using K-Means"
)

plt.xlabel(
    "Annual Income (k$)"
)

plt.ylabel(
    "Spending Score (1-100)"
)

plt.legend(
    title="Cluster"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "customer_clusters.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# -----------------------------
# 19. CLUSTER PROFILING
# -----------------------------

print("\n[14] Creating Cluster Profiles")

cluster_profile = df.groupby(
    "Cluster"
).agg(
    {
        "Age": "mean",
        "Annual Income (k$)": "mean",
        "Spending Score (1-100)": "mean",
        "CustomerID": "count"
    }
)

cluster_profile = cluster_profile.rename(
    columns={
        "CustomerID": "Customer Count"
    }
)

cluster_profile = cluster_profile.round(2)

print("\nCluster Profile:")
print(cluster_profile)


# Save cluster profile
cluster_profile.to_csv(
    OUTPUT_DIR / "cluster_profile.csv"
)


# -----------------------------
# 20. SAVE FINAL SEGMENTED DATA
# -----------------------------

print("\n[15] Saving Final Results")

df.to_csv(
    OUTPUT_DIR / "customer_segments.csv",
    index=False
)

print(
    "Segmented customer data saved successfully."
)


# -----------------------------
# 21. FINAL MODEL INFORMATION
# -----------------------------

print("\n[16] Final Model Information")

print(
    f"Number of clusters : {OPTIMAL_K}"
)

print(
    f"Model inertia      : {kmeans.inertia_:.2f}"
)

print(
    f"Silhouette score   : "
    f"{silhouette_scores[OPTIMAL_K]:.4f}"
)


# -----------------------------
# 22. PROJECT OUTPUTS
# -----------------------------

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated files:")

print("1. income_distribution.png")
print("2. spending_score_distribution.png")
print("3. customers_before_clustering.png")
print("4. elbow_curve.png")
print("5. customer_clusters.png")
print("6. cluster_centers.csv")
print("7. cluster_profile.csv")
print("8. customer_segments.csv")

print("\nAll files are available inside the 'outputs' folder.")

print("=" * 60)


