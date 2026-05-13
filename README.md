# Customer Segmentation Project

A clean Python project for customer segmentation using clustering.

## What this project does

- Generates or loads customer data
- Prepares behavioral and demographic features
- Finds customer segments with K-Means clustering
- Selects the best cluster count when needed
- Creates clear visualizations for the segments
- Saves the segmented dataset and charts for reporting

## Features used

- Age
- Annual income
- Visits per month
- Average order value
- Online purchase rate
- Discount sensitivity
- Loyalty years
- Family size

## Project structure

- `main.py` - command-line entry point
- `src/customer_segmentation.py` - segmentation logic
- `requirements.txt` - Python dependencies
- `outputs/` - generated charts and result files

## Setup

1. Create a virtual environment.
2. Install the dependencies.

```bash
pip install -r requirements.txt
```

## Run the project

Run with the built-in demo dataset:

```bash
python main.py
```

Run with your own CSV file:

```bash
python main.py --input path/to/customer_data.csv
```

Use automatic cluster selection:

```bash
python main.py --clusters auto
```

## Expected output

The script creates:

- `outputs/customer_segments.csv`
- `outputs/cluster_profile.png`
- `outputs/customer_segments_pca.png`
- `outputs/cluster_sizes.png`
- `outputs/elbow_curve.png`

## Using your own data

Your CSV should include these columns:

- `age`
- `annual_income`
- `visits_per_month`
- `avg_order_value`
- `online_purchase_rate`
- `discount_sensitivity`
- `loyalty_years`
- `family_size`

You can also include optional columns like `customer_id`, `gender`, `region`, or `total_spend`.

## How to push to GitHub

```bash




git remote add origin https://github.com/your-username/your-repo-name.git
git push -u origin main
```

## Notes

- The project works even if you do not have a CSV file yet.
- If no input file is provided, the script creates a realistic sample dataset.
- You can replace the sample data with real customer records later.
