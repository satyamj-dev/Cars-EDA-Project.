"""
Exploratory Data Analysis on a Cars Dataset
===========================================
Dataset : "Vehicle dataset from CarDekho" (file: Car details v3.csv), Kaggle.
          Used-car listings scraped from CarDekho (India).
Author  : <Your Name>   |   Batch: <Your Batch>

How to run:
    1. Put "Car details v3.csv" in the same folder as this file
       (or change DATA_PATH below).
    2. pip install pandas numpy matplotlib seaborn
    3. python StudentName_Cars_EDA_Project.py
"""

# =============================================================================
# 0. SETUP
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

DATA_PATH = "Car details v3.csv"   # path to the dataset
SAVE_FIGS = True                   # save every chart as a PNG in ./figures
FIG_DIR = "figures"
REFERENCE_YEAR = 2020              # latest year in the data; used to compute car age

try:
    sns.set_theme(style="whitegrid", context="notebook")
except AttributeError:          # very old seaborn versions
    sns.set(style="whitegrid")

if SAVE_FIGS:
    try:
        os.makedirs(FIG_DIR, exist_ok=True)
    except OSError:             # read-only folder (some online compilers)
        SAVE_FIGS = False


def find_dataset(path):
    """Look for the CSV next to the script, in the current folder, or ask for an upload (Colab)."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))   # __file__ is missing in notebooks
    except NameError:
        here = os.getcwd()
    for folder in (os.getcwd(), here, "/content", "/mnt/data"):
        candidate = os.path.join(folder, path)
        if os.path.exists(candidate):
            return candidate
    try:                        # Google Colab: let the user upload the file
        from google.colab import files
        print(f"'{path}' not found - please upload it now.")
        files.upload()
        return path
    except ImportError:
        raise FileNotFoundError(
            f"Could not find '{path}'. Upload the CSV to the same folder as this script "
            f"(or change DATA_PATH at the top of the file).")


def spearman(frame):
    """Spearman correlation = Pearson correlation of the ranks (no SciPy needed)."""
    return frame.rank().corr()


def finish_chart(fig, filename):
    """Tidy layout, optionally save, then show the chart."""
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIG_DIR, filename), dpi=200, bbox_inches="tight")
    plt.show()


def lakh(x, _pos=None):
    """Axis formatter: show rupees in lakh (1 lakh = 100,000)."""
    return f"{x / 1e5:.0f}"


# =============================================================================
# 1. LOAD DATA AND FIRST INSPECTION
# =============================================================================
raw = pd.read_csv(find_dataset(DATA_PATH))

print("=" * 70)
print("1. DATASET OVERVIEW")
print("=" * 70)
print("Shape (rows, columns):", raw.shape)
print("\nColumn names and data types:")
print(raw.dtypes)
print("\nSample rows:")
print(raw.head(5).to_string())

numeric_raw = raw.select_dtypes(include="number").columns.tolist()
categorical_raw = raw.select_dtypes(exclude="number").columns.tolist()
print("\nNumerical columns  :", numeric_raw)
print("Categorical columns:", categorical_raw)

# =============================================================================
# 2. DATA QUALITY CHECKS (before cleaning)
# =============================================================================
print("\n" + "=" * 70)
print("2. DATA QUALITY CHECKS")
print("=" * 70)
rows_before = len(raw)
print("Missing values per column:")
print(raw.isna().sum()[raw.isna().sum() > 0])
print("Rows with at least one missing value:", int(raw.isna().any(axis=1).sum()))
print("Exact duplicate rows:", int(raw.duplicated().sum()))

print("\nCategory labels (checking for inconsistent spelling/case):")
for col in ["fuel", "seller_type", "transmission", "owner"]:
    print(f"\n{col}:")
    print(raw[col].value_counts(dropna=False))

print("\nExample of text-formatted numeric columns:")
print(raw[["mileage", "engine", "max_power", "torque"]].head(3).to_string())
print("Mileage units present:",
      raw["mileage"].dropna().str.extract(r"([A-Za-z/]+)$")[0].value_counts().to_dict())

# =============================================================================
# 3. DATA CLEANING
# =============================================================================
print("\n" + "=" * 70)
print("3. DATA CLEANING")
print("=" * 70)
df = raw.copy()

# 3.1 Remove exact duplicate rows (same listing scraped more than once)
n_dups = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print(f"3.1 Dropped {n_dups} exact duplicate rows -> {len(df)} rows")

# 3.2 Clean text labels
for col in ["fuel", "seller_type", "transmission", "owner"]:
    df[col] = df[col].str.strip()
print("3.2 Stripped whitespace from category columns")

# 3.3 Convert text columns into numbers (e.g. '23.4 kmpl' -> 23.4)
mileage_unit = df["mileage"].str.extract(r"([A-Za-z/]+)$")[0]
df["mileage_value"] = pd.to_numeric(df["mileage"].str.extract(r"([\d.]+)")[0], errors="coerce")
# km/kg (CNG/LPG) and kmpl (petrol/diesel) are different units -> keep only kmpl
df["mileage_kmpl"] = df["mileage_value"].where(mileage_unit == "kmpl")
df["engine_cc"] = pd.to_numeric(df["engine"].str.extract(r"([\d.]+)")[0], errors="coerce")
df["max_power_bhp"] = pd.to_numeric(df["max_power"].str.extract(r"([\d.]+)")[0], errors="coerce")
df.loc[df["max_power_bhp"] == 0, "max_power_bhp"] = np.nan   # 0 bhp is impossible
n_zero_mileage = int((df["mileage_kmpl"] == 0).sum())
df.loc[df["mileage_kmpl"] == 0, "mileage_kmpl"] = np.nan     # 0 kmpl is impossible
print(f"    ({n_zero_mileage} cars with 0 kmpl mileage, and any 0 bhp power values, treated as missing)")
n_kmkg = int((mileage_unit == "km/kg").sum())
print(f"3.3 Converted mileage/engine/max_power to numbers. {n_kmkg} CNG/LPG cars "
      f"report mileage in km/kg, so they are excluded from mileage_kmpl.")

# 3.4 Drop columns not needed (torque has mixed units/formats; originals replaced)
df = df.drop(columns=["torque", "mileage", "engine", "max_power", "mileage_value"])
print("3.4 Dropped 'torque' (inconsistent units) and the original text columns")

# 3.5 Feature engineering: brand and car age
df["brand"] = df["name"].str.split().str[0]
df["brand"] = df["brand"].replace({"Land": "Land Rover", "Ashok": "Ashok Leyland"})
df["car_age"] = REFERENCE_YEAR - df["year"]
print("3.5 Created 'brand' (from name) and 'car_age' (2020 - year)")

# 3.6 Impossible / extreme values: km_driven above 500,000 km or below 100 km
extreme_km = (df["km_driven"] > 500_000) | (df["km_driven"] < 100)
print(f"3.6 Removing {int(extreme_km.sum())} rows with implausible km_driven "
      f"(> 500,000 or < 100): {sorted(df.loc[extreme_km, 'km_driven'].tolist(), reverse=True)}")
df = df[~extreme_km].reset_index(drop=True)

# 3.7 Missing values: fill with brand median (engine, power), fuel median (mileage), mode (seats)
missing_before_impute = df[["mileage_kmpl", "engine_cc", "max_power_bhp", "seats"]].isna().sum()
print("\n3.7 Missing values before filling (mileage_kmpl also counts the CNG/LPG cars that use km/kg):")
print(missing_before_impute)
for col in ["engine_cc", "max_power_bhp"]:
    df[col] = df[col].fillna(df.groupby("brand")[col].transform("median"))
    df[col] = df[col].fillna(df[col].median())
# mileage: fill by fuel-type median (CNG/LPG cars stay NaN because they use km/kg)
df["mileage_kmpl"] = df["mileage_kmpl"].fillna(df.groupby("fuel")["mileage_kmpl"].transform("median"))
df["seats"] = df["seats"].fillna(df["seats"].mode()[0]).astype(int)
print("Missing after filling:")
print(df[["mileage_kmpl", "engine_cc", "max_power_bhp", "seats"]].isna().sum())

# 3.8 Final cleaned dataset summary
rows_after = len(df)
print(f"\nDataset size BEFORE cleaning: {rows_before} rows x {raw.shape[1]} columns")
print(f"Dataset size AFTER cleaning : {rows_after} rows x {df.shape[1]} columns")
print(f"Rows removed: {rows_before - rows_after} ({(rows_before - rows_after) / rows_before:.1%})")

num_cols = ["selling_price", "year", "car_age", "km_driven", "mileage_kmpl",
            "engine_cc", "max_power_bhp", "seats"]
cat_cols = ["brand", "fuel", "seller_type", "transmission", "owner"]

# =============================================================================
# 4. OUTLIER CHECK (IQR method)
# =============================================================================
print("\n" + "=" * 70)
print("4. OUTLIERS (IQR rule: below Q1-1.5*IQR or above Q3+1.5*IQR)")
print("=" * 70)
for col in ["selling_price", "km_driven", "engine_cc", "max_power_bhp"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((df[col] < low) | (df[col] > high)).sum())
    print(f"{col:15s}: {n_out:4d} outliers ({n_out / len(df):.1%}); upper fence = {high:,.0f}")
print("\nTop 5 most expensive listings (kept: genuine luxury cars):")
print(df.nlargest(5, "selling_price")[["name", "year", "selling_price", "km_driven"]].to_string(index=False))

# =============================================================================
# 5. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "=" * 70)
print("5. DESCRIPTIVE STATISTICS")
print("=" * 70)
print("Numerical columns:")
print(df[num_cols].describe().round(1).T.to_string())
print("\nSkewness of selling_price:", round(df["selling_price"].skew(), 2),
      "| skewness of log(price):", round(np.log(df["selling_price"]).skew(), 2))
print("\nCategorical columns:")
for col in ["fuel", "seller_type", "transmission", "owner"]:
    counts = df[col].value_counts()
    print(f"\n{col}:")
    print(pd.DataFrame({"count": counts, "percent": (counts / len(df) * 100).round(1)}))
print("\nNumber of unique brands:", df["brand"].nunique())
top_brands = df["brand"].value_counts().head(10)
print(pd.DataFrame({"count": top_brands, "percent": (top_brands / len(df) * 100).round(1)}))

# =============================================================================
# 6. VISUAL EDA
# =============================================================================
df["log_price"] = np.log10(df["selling_price"])

# --- Chart 1: bar chart - most common brands (univariate, categorical) -------
fig, ax = plt.subplots(figsize=(8, 3.6))
sns.barplot(x=top_brands.values, y=top_brands.index, color="#3b78c3", ax=ax)
ax.set_title("Chart 1: Ten Most Common Car Brands in the Dataset")
ax.set_xlabel("Number of listings")
ax.set_ylabel("Brand")
for i, v in enumerate(top_brands.values):
    ax.text(v + 15, i, str(v), va="center", fontsize=9)
finish_chart(fig, "chart1_brand_counts.png")

# --- Chart 2: histogram - price distribution (univariate, numerical) ---------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
sns.histplot(df["selling_price"], bins=50, color="#3b78c3", ax=axes[0])
axes[0].set_title("Selling price (raw)")
axes[0].set_xlabel("Selling price (Rs. lakh)")
axes[0].xaxis.set_major_formatter(FuncFormatter(lakh))
sns.histplot(df["log_price"], bins=40, color="#e08a1e", ax=axes[1])
axes[1].set_title("Selling price (log10 scale)")
axes[1].set_xlabel("log10(selling price in Rs.)")
fig.suptitle("Chart 2: Distribution of Selling Price", fontsize=13, y=1.02)
finish_chart(fig, "chart2_price_hist.png")

# --- Chart 3: box plots - price by fuel type and by transmission -------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
fuel_order = df["fuel"].value_counts().index
sns.boxplot(data=df, x="fuel", y="log_price", order=fuel_order, palette="Set2", ax=axes[0])
axes[0].set_title("By fuel type")
axes[0].set_xlabel("Fuel type")
axes[0].set_ylabel("log10(selling price in Rs.)")
sns.boxplot(data=df, x="transmission", y="log_price", palette="Set2", ax=axes[1])
axes[1].set_title("By transmission")
axes[1].set_xlabel("Transmission")
axes[1].set_ylabel("log10(selling price in Rs.)")
fig.suptitle("Chart 3: Selling Price by Fuel Type and Transmission", fontsize=13, y=1.02)
finish_chart(fig, "chart3_price_boxplots.png")

# --- Chart 4: scatter - car age vs price (bivariate) -------------------------
fig, ax = plt.subplots(figsize=(8, 3.8))
sns.scatterplot(data=df, x="car_age", y="log_price", alpha=0.25, s=14,
                color="#3b78c3", edgecolor=None, ax=ax)
median_by_age = df.groupby("car_age")["log_price"].median()
ax.plot(median_by_age.index, median_by_age.values, color="#d62728", lw=2, label="Median price per age")
ax.set_title("Chart 4: Car Age vs Selling Price")
ax.set_xlabel("Car age (years, 2020 - manufacturing year)")
ax.set_ylabel("log10(selling price in Rs.)")
ax.legend()
finish_chart(fig, "chart4_age_vs_price.png")

# --- Chart 5: scatter - km driven vs price (bivariate) -----------------------
fig, ax = plt.subplots(figsize=(8, 3.8))
sns.scatterplot(data=df, x="km_driven", y="selling_price", hue="transmission",
                alpha=0.35, s=14, edgecolor=None, ax=ax)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("Chart 5: Kilometres Driven vs Selling Price (log-log scale)")
ax.set_xlabel("Kilometres driven (log scale)")
ax.set_ylabel("Selling price in Rs. (log scale)")
finish_chart(fig, "chart5_km_vs_price.png")

# --- Chart 6: bar chart - median price for brands with 30+ listings ----------
brand_stats = (df.groupby("brand")["selling_price"].agg(["median", "mean", "count"])
                 .query("count >= 30").sort_values("median", ascending=False))
fig, ax = plt.subplots(figsize=(8, 4.4))
sns.barplot(x=brand_stats["median"] / 1e5, y=brand_stats.index, color="#2a9d8f", ax=ax)
ax.set_title("Chart 6: Median Selling Price by Brand (brands with 30+ listings)")
ax.set_xlabel("Median selling price (Rs. lakh)")
ax.set_ylabel("Brand")
finish_chart(fig, "chart6_brand_median_price.png")

# --- Chart 7: heatmap - correlation matrix (multivariate) --------------------
corr_cols = ["selling_price", "car_age", "km_driven", "mileage_kmpl",
             "engine_cc", "max_power_bhp", "seats"]
corr = spearman(df[corr_cols])
fig, ax = plt.subplots(figsize=(7, 5.2))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            linewidths=0.5, ax=ax)
ax.set_title("Chart 7: Spearman Correlation Heatmap of Numerical Features")
finish_chart(fig, "chart7_correlation_heatmap.png")

# =============================================================================
# 7. FINDINGS: numbers used to support the written insights
# =============================================================================
print("\n" + "=" * 70)
print("7. NUMBERS BEHIND THE KEY FINDINGS")
print("=" * 70)

# F1: brand concentration
top2 = df["brand"].value_counts().head(2)
print(f"F1 Top two brands: {top2.to_dict()} -> {top2.sum() / len(df):.1%} of listings")
print(f"   Fuel share: {(df['fuel'].value_counts(normalize=True) * 100).round(1).to_dict()}")

# F2: price skew
print(f"F2 Price mean = Rs.{df['selling_price'].mean():,.0f}, median = Rs.{df['selling_price'].median():,.0f}, "
      f"max = Rs.{df['selling_price'].max():,.0f}")

# F3: age vs price
print("F3 Spearman(car_age, price) =", round(spearman(df[['car_age', 'selling_price']]).iloc[0, 1], 2))
age_groups = pd.cut(df["car_age"], [-1, 2, 5, 10, 15, 40], labels=["0-2", "3-5", "6-10", "11-15", "16+"])
print("   Median price by age group (years):")
print(df.groupby(age_groups, observed=True)["selling_price"].agg(["median", "count"]).to_string())

# F4: km vs price, and km vs age
print("F4 Spearman(km_driven, price) =", round(spearman(df[['km_driven', 'selling_price']]).iloc[0, 1], 2),
      "| Spearman(km_driven, car_age) =", round(spearman(df[['km_driven', 'car_age']]).iloc[0, 1], 2))

# F5: transmission
trans = df.groupby("transmission").agg(median_price=("selling_price", "median"),
                                       median_power=("max_power_bhp", "median"),
                                       median_age=("car_age", "median"),
                                       count=("selling_price", "size"))
print("F5 Transmission comparison:")
print(trans.to_string())

# F6: fuel
fuel_tbl = df.groupby("fuel").agg(median_price=("selling_price", "median"),
                                  median_engine=("engine_cc", "median"),
                                  median_age=("car_age", "median"),
                                  count=("selling_price", "size"))
print("F6 Fuel comparison:")
print(fuel_tbl.to_string())

# F7: brand prices
print("F7 Median price by brand (30+ listings), top 5 and bottom 3:")
print(brand_stats[["median", "mean", "count"]].head(5).round(0).to_string())
print(brand_stats[["median", "mean", "count"]].tail(3).round(0).to_string())

# F8: correlations with price and power/engine
print("F8 Spearman correlation with selling_price:")
print(corr["selling_price"].drop("selling_price").round(2).to_string())

# F9: owner type and seller type
print("F9 Median price by owner / seller type:")
print(df.groupby("owner")["selling_price"].agg(["median", "count"]).to_string())
print(df.groupby("seller_type")["selling_price"].agg(["median", "count"]).to_string())

print("\nDone. Charts saved in:", os.path.abspath(FIG_DIR) if SAVE_FIGS else "(not saved)")
