from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "age",
    "annual_income",
    "visits_per_month",
    "avg_order_value",
    "online_purchase_rate",
    "discount_sensitivity",
    "loyalty_years",
    "family_size",
]

OPTIONAL_COLUMNS = ["customer_id", "gender", "region", "total_spend", "latent_segment"]
DEFAULT_RANDOM_STATE = 42
DEFAULT_CLUSTER_COUNT = 4


@dataclass(frozen=True)
class AnalysisResult:
    data: pd.DataFrame
    feature_columns: list[str]
    cluster_count: int
    silhouette: float
    profile: pd.DataFrame
    output_dir: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Customer segmentation with K-Means clustering."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional path to a customer CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where charts and results will be saved.",
    )
    parser.add_argument(
        "--clusters",
        type=str,
        default=str(DEFAULT_CLUSTER_COUNT),
        help='Number of clusters to use, or "auto" to choose the best silhouette score.',
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for reproducible results.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=240,
        help="Number of rows to generate when no input file is provided.",
    )
    return parser


def generate_sample_customer_data(sample_size: int = 240, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    segment_profiles = [
        {
            "name": "Premium Loyalists",
            "share": 0.25,
            "age": (44, 6),
            "annual_income": (92000, 9000),
            "visits_per_month": (8.0, 1.4),
            "avg_order_value": (185.0, 22.0),
            "online_purchase_rate": (0.58, 0.10),
            "discount_sensitivity": (0.28, 0.08),
            "loyalty_years": (6.5, 1.4),
            "family_size": (3.0, 1.0),
        },
        {
            "name": "Digital Explorers",
            "share": 0.30,
            "age": (32, 7),
            "annual_income": (68000, 8500),
            "visits_per_month": (11.5, 2.0),
            "avg_order_value": (125.0, 18.0),
            "online_purchase_rate": (0.88, 0.07),
            "discount_sensitivity": (0.46, 0.10),
            "loyalty_years": (3.5, 1.2),
            "family_size": (2.0, 1.0),
        },
        {
            "name": "Value Seekers",
            "share": 0.25,
            "age": (39, 8),
            "annual_income": (47000, 6500),
            "visits_per_month": (6.5, 1.8),
            "avg_order_value": (82.0, 14.0),
            "online_purchase_rate": (0.62, 0.12),
            "discount_sensitivity": (0.82, 0.08),
            "loyalty_years": (4.0, 1.3),
            "family_size": (4.0, 1.0),
        },
        {
            "name": "Occasional Shoppers",
            "share": 0.20,
            "age": (53, 7),
            "annual_income": (73000, 10000),
            "visits_per_month": (3.0, 1.0),
            "avg_order_value": (145.0, 20.0),
            "online_purchase_rate": (0.36, 0.10),
            "discount_sensitivity": (0.30, 0.09),
            "loyalty_years": (2.0, 1.0),
            "family_size": (2.0, 1.0),
        },
    ]

    records: list[dict[str, object]] = []
    customer_counter = 1

    for profile in segment_profiles:
        row_count = max(1, int(round(sample_size * profile["share"])))
        for _ in range(row_count):
            age = int(np.clip(rng.normal(*profile["age"]), 18, 70))
            annual_income = int(np.clip(rng.normal(*profile["annual_income"]), 22000, 160000))
            visits_per_month = round(float(np.clip(rng.normal(*profile["visits_per_month"]), 1, 20)), 1)
            avg_order_value = round(float(np.clip(rng.normal(*profile["avg_order_value"]), 20, 400)), 2)
            online_purchase_rate = round(float(np.clip(rng.normal(*profile["online_purchase_rate"]), 0.05, 0.98)), 2)
            discount_sensitivity = round(float(np.clip(rng.normal(*profile["discount_sensitivity"]), 0.05, 0.98)), 2)
            loyalty_years = round(float(np.clip(rng.normal(*profile["loyalty_years"]), 0.0, 15.0)), 1)
            family_size = int(np.clip(round(rng.normal(*profile["family_size"])), 1, 7))

            seasonality = round(float(np.clip(rng.normal(0.8 + visits_per_month / 25.0, 0.1), 0.5, 1.5)), 2)
            total_spend = round(visits_per_month * avg_order_value * 12 * seasonality, 2)

            records.append(
                {
                    "customer_id": f"C{customer_counter:04d}",
                    "age": age,
                    "gender": rng.choice(["Female", "Male", "Other"], p=[0.46, 0.50, 0.04]),
                    "region": rng.choice(["North", "South", "East", "West"]),
                    "annual_income": annual_income,
                    "visits_per_month": visits_per_month,
                    "avg_order_value": avg_order_value,
                    "online_purchase_rate": online_purchase_rate,
                    "discount_sensitivity": discount_sensitivity,
                    "loyalty_years": loyalty_years,
                    "family_size": family_size,
                    "total_spend": total_spend,
                    "latent_segment": profile["name"],
                }
            )
            customer_counter += 1

    data = pd.DataFrame.from_records(records)
    data = data.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    data["customer_id"] = [f"C{i:04d}" for i in range(1, len(data) + 1)]
    return data


def load_customer_data(input_path: str | None, sample_size: int, random_state: int) -> pd.DataFrame:
    if input_path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return pd.read_csv(path)

    return generate_sample_customer_data(sample_size=sample_size, random_state=random_state)


def validate_columns(data: pd.DataFrame) -> None:
    missing_columns = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )


def choose_cluster_count(scaled_features: np.ndarray, random_state: int) -> tuple[int, list[dict[str, float]]]:
    scores: list[dict[str, float]] = []
    best_k = 4
    best_score = -1.0

    max_k = min(8, len(scaled_features) - 1)
    min_k = 2
    if max_k < min_k:
        return 1, scores

    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(scaled_features)
        score = silhouette_score(scaled_features, labels)
        inertia = float(model.inertia_)
        scores.append({"k": float(k), "silhouette": float(score), "inertia": inertia})
        if score > best_score:
            best_k = k
            best_score = score

    return best_k, scores


def summarize_segments(data: pd.DataFrame, cluster_column: str) -> pd.DataFrame:
    summary = (
        data.groupby(cluster_column)[FEATURE_COLUMNS]
        .mean()
        .round(2)
        .sort_index()
    )
    summary["customer_count"] = data.groupby(cluster_column).size()
    return summary


def label_segments(summary: pd.DataFrame) -> dict[int, str]:
    remaining = set(summary.index.astype(int))
    label_map: dict[int, str] = {}

    premium_score = summary[["annual_income", "avg_order_value", "loyalty_years"]].sum(axis=1)
    premium_cluster = int(premium_score.idxmax())
    label_map[premium_cluster] = "Premium Loyalists"
    remaining.discard(premium_cluster)

    if remaining:
        digital_score = summary.loc[list(remaining), ["online_purchase_rate", "visits_per_month"]].sum(axis=1)
        digital_cluster = int(digital_score.idxmax())
        label_map[digital_cluster] = "Digital Explorers"
        remaining.discard(digital_cluster)

    if remaining:
        value_score = summary.loc[list(remaining), ["discount_sensitivity"]].sum(axis=1)
        value_cluster = int(value_score.idxmax())
        label_map[value_cluster] = "Value Seekers"
        remaining.discard(value_cluster)

    for cluster_id in sorted(remaining):
        label_map[cluster_id] = "Occasional Shoppers"

    return label_map


def apply_segment_names(data: pd.DataFrame, cluster_column: str, summary: pd.DataFrame) -> pd.DataFrame:
    label_map = label_segments(summary)
    result = data.copy()
    result["segment_name"] = result[cluster_column].map(label_map).fillna("Customer Segment")
    return result


def fit_clustering(data: pd.DataFrame, requested_clusters: str, random_state: int) -> tuple[pd.DataFrame, int, float, list[dict[str, float]]]:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[FEATURE_COLUMNS])

    if requested_clusters.lower() == "auto":
        cluster_count, evaluation_scores = choose_cluster_count(scaled_features, random_state)
    else:
        cluster_count = int(requested_clusters)
        evaluation_scores = []

    if cluster_count < 2:
        cluster_count = 2

    model = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=10)
    labels = model.fit_predict(scaled_features)
    silhouette = silhouette_score(scaled_features, labels)

    clustered = data.copy()
    clustered["segment_id"] = labels
    return clustered, cluster_count, float(silhouette), evaluation_scores


def save_elbow_curve(scaled_features: np.ndarray, output_dir: Path, random_state: int) -> None:
    inertias: list[float] = []
    cluster_values = list(range(2, min(9, len(scaled_features))))

    for cluster_count in cluster_values:
        model = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=10)
        model.fit(scaled_features)
        inertias.append(float(model.inertia_))

    plt.figure(figsize=(8, 5))
    plt.plot(cluster_values, inertias, marker="o", color="#0f766e", linewidth=2)
    plt.title("Elbow Curve")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(output_dir / "elbow_curve.png", dpi=160)
    plt.close()


def save_segment_profile_chart(summary: pd.DataFrame, output_dir: Path) -> None:
    heatmap_data = summary[FEATURE_COLUMNS].copy()
    scaler = StandardScaler()
    scaled_profile = pd.DataFrame(
        scaler.fit_transform(heatmap_data),
        index=heatmap_data.index.astype(str),
        columns=heatmap_data.columns,
    )

    plt.figure(figsize=(12, 6))
    image = plt.imshow(scaled_profile.values, aspect="auto", cmap="YlGnBu")
    plt.colorbar(image, label="Relative feature level")
    plt.xticks(range(len(scaled_profile.columns)), scaled_profile.columns, rotation=30, ha="right")
    plt.yticks(range(len(scaled_profile.index)), scaled_profile.index)
    plt.title("Cluster Profile Overview")
    plt.xlabel("Features")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_profile.png", dpi=160)
    plt.close()


def save_cluster_sizes(data: pd.DataFrame, output_dir: Path) -> None:
    counts = data["segment_name"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(counts)))
    plt.barh(counts.index, counts.values, color=colors)
    plt.title("Customer Count by Segment")
    plt.xlabel("Customers")
    plt.ylabel("Segment")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_sizes.png", dpi=160)
    plt.close()


def save_pca_scatter(data: pd.DataFrame, output_dir: Path) -> None:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[FEATURE_COLUMNS])
    pca = PCA(n_components=2, random_state=DEFAULT_RANDOM_STATE)
    components = pca.fit_transform(scaled_features)

    plot_data = pd.DataFrame(
        {
            "pc1": components[:, 0],
            "pc2": components[:, 1],
            "segment_name": data["segment_name"],
        }
    )

    plt.figure(figsize=(9, 6))
    unique_segments = list(plot_data["segment_name"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, max(1, len(unique_segments))))
    for color, segment_name in zip(palette, unique_segments):
        segment_points = plot_data[plot_data["segment_name"] == segment_name]
        plt.scatter(
            segment_points["pc1"],
            segment_points["pc2"],
            label=segment_name,
            s=70,
            alpha=0.9,
            color=color,
            edgecolors="white",
            linewidths=0.4,
        )
    plt.title("Customer Segments in PCA Space")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Segment", loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "customer_segments_pca.png", dpi=160)
    plt.close()


def save_results(data: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "customer_segments.csv", index=False)


def print_report(result: AnalysisResult) -> None:
    print("\nCustomer Segmentation Summary")
    print("=" * 32)
    print(f"Rows analyzed: {len(result.data)}")
    print(f"Clusters used: {result.cluster_count}")
    print(f"Silhouette score: {result.silhouette:.4f}")
    print("\nSegment profile:")
    for segment_name, row in result.profile.iterrows():
        print(
            f"- {segment_name}: customers={int(row['customer_count'])}, "
            f"age={row['age']:.1f}, income={row['annual_income']:.0f}, "
            f"visits={row['visits_per_month']:.1f}, order={row['avg_order_value']:.2f}, "
            f"online={row['online_purchase_rate']:.2f}, discount={row['discount_sensitivity']:.2f}, "
            f"loyalty={row['loyalty_years']:.1f}, family={row['family_size']:.1f}"
        )
    print(f"\nFiles saved to: {result.output_dir.resolve()}")


def run_analysis(input_path: str | None, output_dir: str, clusters: str, sample_size: int, random_state: int) -> AnalysisResult:
    data = load_customer_data(input_path=input_path, sample_size=sample_size, random_state=random_state)
    validate_columns(data)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    clustered_data, cluster_count, silhouette, _ = fit_clustering(
        data=data,
        requested_clusters=clusters,
        random_state=random_state,
    )
    summary = summarize_segments(clustered_data, "segment_id")
    labeled_data = apply_segment_names(clustered_data, "segment_id", summary)
    labeled_summary = summarize_segments(labeled_data, "segment_name")
    save_results(labeled_data, output_path)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(labeled_data[FEATURE_COLUMNS])
    save_elbow_curve(scaled_features, output_path, random_state)
    save_segment_profile_chart(summary, output_path)
    save_cluster_sizes(labeled_data, output_path)
    save_pca_scatter(labeled_data, output_path)

    return AnalysisResult(
        data=labeled_data,
        feature_columns=FEATURE_COLUMNS,
        cluster_count=cluster_count,
        silhouette=silhouette,
        profile=labeled_summary,
        output_dir=output_path,
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        clusters=args.clusters,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
