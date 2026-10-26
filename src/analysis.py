"""
E-Commerce Business Intelligence & Customer Analytics
-----------------------------------------------------
Sections: 1) KPIs  2) Sales trends  3) Product/category  4) Geography & channel
          5) RFM segmentation  6) Cohort retention  7) CLV  8) K-Means clustering
Run: python src/analysis.py   (after generate_data.py)
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DATA, IMG, OUT = ROOT / "data", ROOT / "images", ROOT / "outputs"
for p in (IMG, OUT):
    p.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = sns.color_palette("viridis", 8)
inr = lambda x: f"₹{x:,.0f}"

# ------------------------------------------------------------------ load
customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
products = pd.read_csv(DATA / "products.csv")
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
items = pd.read_csv(DATA / "order_items.csv")

items["net_revenue"] = items.unit_price * items.quantity * (1 - items.discount)
items = items.merge(products[["product_id", "category", "cost"]], on="product_id")
items["profit"] = items.net_revenue - items.cost * items.quantity
items = items.merge(orders[["order_id", "customer_id", "order_date", "order_status"]], on="order_id")

valid_orders = orders[orders.order_status == "Delivered"].copy()     # revenue-recognised orders
valid_items = items[items.order_status == "Delivered"].copy()
valid_orders["month"] = valid_orders.order_date.dt.to_period("M")
valid_items["month"] = valid_items.order_date.dt.to_period("M")

insights = {}

# ------------------------------------------------------------------ 1. KPIs
total_rev = valid_orders.order_value.sum()
total_profit = valid_items.profit.sum()
n_orders = len(valid_orders)
n_cust = valid_orders.customer_id.nunique()
aov = total_rev / n_orders
orders_per_cust = valid_orders.groupby("customer_id").size()
repeat_rate = (orders_per_cust > 1).mean()
return_rate = (orders.order_status == "Returned").mean()
cancel_rate = (orders.order_status == "Cancelled").mean()
kpis = {
    "Total Revenue": inr(total_rev), "Total Profit": inr(total_profit),
    "Profit Margin %": round(total_profit / total_rev * 100, 1),
    "Delivered Orders": f"{n_orders:,}", "Active Customers": f"{n_cust:,}",
    "Average Order Value": inr(aov), "Repeat Customer Rate %": round(repeat_rate * 100, 1),
    "Return Rate %": round(return_rate * 100, 1), "Cancellation Rate %": round(cancel_rate * 100, 1),
}
insights["kpis"] = kpis
pd.Series(kpis).to_csv(OUT / "kpi_summary.csv", header=["value"])

# ------------------------------------------------------------------ 2. Sales trend
monthly = valid_orders.groupby("month").agg(revenue=("order_value", "sum"), orders=("order_id", "count"),
                                            customers=("customer_id", "nunique"))
monthly["aov"] = monthly.revenue / monthly.orders
monthly["mom_growth_%"] = monthly.revenue.pct_change() * 100
monthly.to_csv(OUT / "monthly_sales.csv")
monthly.index = monthly.index.to_timestamp()

fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax[0].plot(monthly.index, monthly.revenue / 1e5, marker="o", color=PALETTE[2], lw=2)
ax[0].fill_between(monthly.index, monthly.revenue / 1e5, alpha=.15, color=PALETTE[2])
ax[0].set_title("Monthly Revenue (₹ Lakhs)", fontweight="bold"); ax[0].set_ylabel("₹ Lakhs")
ax[1].bar(monthly.index, monthly.orders, width=20, color=PALETTE[5])
ax[1].set_title("Monthly Orders", fontweight="bold"); ax[1].set_ylabel("Orders")
plt.tight_layout(); plt.savefig(IMG / "01_monthly_revenue_orders.png", dpi=150); plt.close()

peak = monthly.revenue.idxmax()
insights["peak_month"] = [peak.strftime("%b %Y"), inr(monthly.revenue.max())]
festive_months = valid_orders[valid_orders.order_date.dt.month.isin([10, 11])].groupby(valid_orders.order_date.dt.year).order_value.sum()
insights["oct_nov_revenue_by_year"] = {int(k): inr(v) for k, v in festive_months.items()}

# ------------------------------------------------------------------ 3. Category / product
cat = valid_items.groupby("category").agg(revenue=("net_revenue", "sum"), profit=("profit", "sum"),
                                           units=("quantity", "sum"))
cat["margin_%"] = cat.profit / cat.revenue * 100
cat = cat.sort_values("revenue", ascending=False)
cat.to_csv(OUT / "category_performance.csv")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(x=cat.revenue / 1e5, y=cat.index, ax=ax[0], palette="viridis")
ax[0].set_title("Revenue by Category (₹ Lakhs)", fontweight="bold"); ax[0].set_xlabel("₹ Lakhs"); ax[0].set_ylabel("")
sns.barplot(x=cat["margin_%"], y=cat.index, ax=ax[1], palette="magma")
ax[1].set_title("Profit Margin % by Category", fontweight="bold"); ax[1].set_xlabel("%"); ax[1].set_ylabel("")
plt.tight_layout(); plt.savefig(IMG / "02_category_performance.png", dpi=150); plt.close()
insights["top_category"] = [cat.index[0], inr(cat.revenue.iloc[0])]
insights["best_margin_category"] = [cat["margin_%"].idxmax(), round(cat["margin_%"].max(), 1)]
insights["lowest_margin_category"] = [cat["margin_%"].idxmin(), round(cat["margin_%"].min(), 1)]

top_products = (valid_items.merge(products[["product_id", "product_name"]], on="product_id")
                .groupby(["product_name", "category"]).net_revenue.sum().nlargest(10).reset_index())
top_products.to_csv(OUT / "top10_products.csv", index=False)

# ------------------------------------------------------------------ 4. Geography, channel, payments
vo = valid_orders.merge(customers, on="customer_id")
city = vo.groupby("city").order_value.sum().sort_values(ascending=False)
channel = vo.groupby("acquisition_channel").agg(revenue=("order_value", "sum"),
                                                customers=("customer_id", "nunique"),
                                                orders=("order_id", "count"))
channel["revenue_per_customer"] = channel.revenue / channel.customers
channel["orders_per_customer"] = channel.orders / channel.customers
channel.to_csv(OUT / "channel_performance.csv")
pay = orders.groupby("payment_method").size().sort_values(ascending=False)
cod_return = orders.assign(ret=orders.order_status.eq("Returned")).groupby("payment_method").ret.mean() * 100

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
sns.barplot(x=city.values / 1e5, y=city.index, ax=ax[0], palette="crest")
ax[0].set_title("Revenue by City (₹ Lakhs)", fontweight="bold"); ax[0].set_ylabel("")
sns.barplot(x=channel.revenue_per_customer, y=channel.index, ax=ax[1], palette="flare")
ax[1].set_title("Revenue per Customer by Channel (₹)", fontweight="bold"); ax[1].set_ylabel("")
ax[2].pie(pay.values, labels=pay.index, autopct="%1.0f%%", colors=sns.color_palette("Set2"), startangle=90)
ax[2].set_title("Payment Method Mix", fontweight="bold")
plt.tight_layout(); plt.savefig(IMG / "03_city_channel_payment.png", dpi=150); plt.close()
insights["top_city"] = [city.index[0], inr(city.iloc[0])]
insights["best_channel_rev_per_customer"] = [channel.revenue_per_customer.idxmax(), inr(channel.revenue_per_customer.max())]
insights["worst_channel_rev_per_customer"] = [channel.revenue_per_customer.idxmin(), inr(channel.revenue_per_customer.min())]
insights["highest_return_payment"] = [cod_return.idxmax(), round(cod_return.max(), 1)]

# ------------------------------------------------------------------ 5. RFM
snapshot = valid_orders.order_date.max() + pd.Timedelta(days=1)
rfm = valid_orders.groupby("customer_id").agg(Recency=("order_date", lambda d: (snapshot - d.max()).days),
                                              Frequency=("order_id", "count"),
                                              Monetary=("order_value", "sum"))
rfm["R"] = pd.qcut(rfm.Recency, 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["F"] = pd.qcut(rfm.Frequency.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["M"] = pd.qcut(rfm.Monetary.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

def segment(r):
    if r.R >= 4 and r.F >= 4:            return "Champions"
    if r.R >= 3 and r.F >= 3:            return "Loyal Customers"
    if r.R >= 4 and r.F <= 2:            return "New / Promising"
    if r.R == 3 and r.F <= 2:            return "Need Attention"
    if r.R <= 2 and r.F >= 4:            return "Can't Lose Them"
    if r.R <= 2 and r.F >= 3:            return "At Risk"
    if r.R <= 2 and r.F <= 2 and r.R == 1: return "Lost"
    return "Hibernating"
rfm["Segment"] = rfm.apply(segment, axis=1)
rfm.to_csv(OUT / "rfm_segments.csv")

seg = rfm.groupby("Segment").agg(customers=("Monetary", "size"), avg_recency=("Recency", "mean"),
                                 avg_frequency=("Frequency", "mean"), revenue=("Monetary", "sum"))
seg["customers_%"] = seg.customers / seg.customers.sum() * 100
seg["revenue_%"] = seg.revenue / seg.revenue.sum() * 100
seg = seg.sort_values("revenue", ascending=False)
seg.to_csv(OUT / "rfm_segment_summary.csv")

fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
sns.barplot(x=seg["customers_%"], y=seg.index, ax=ax[0], palette="viridis")
ax[0].set_title("% of Customers by RFM Segment", fontweight="bold"); ax[0].set_ylabel("")
sns.barplot(x=seg["revenue_%"], y=seg.index, ax=ax[1], palette="plasma")
ax[1].set_title("% of Revenue by RFM Segment", fontweight="bold"); ax[1].set_ylabel("")
plt.tight_layout(); plt.savefig(IMG / "04_rfm_segments.png", dpi=150); plt.close()

insights["rfm"] = {s: {"customers_%": round(r["customers_%"], 1), "revenue_%": round(r["revenue_%"], 1)}
                   for s, r in seg.iterrows()}
top20 = rfm.Monetary.sort_values(ascending=False)
insights["pareto_top20pct_customers_revenue_%"] = round(top20.head(int(len(top20) * .2)).sum() / top20.sum() * 100, 1)

# ------------------------------------------------------------------ 6. Cohort retention
vo2 = valid_orders.merge(customers[["customer_id", "signup_date"]], on="customer_id")
vo2["cohort"] = vo2.signup_date.dt.to_period("M")
vo2["period"] = (vo2.order_date.dt.year - vo2.signup_date.dt.year) * 12 + (vo2.order_date.dt.month - vo2.signup_date.dt.month)
cohort_counts = vo2.groupby(["cohort", "period"]).customer_id.nunique().unstack(fill_value=0)
cohort_size = customers.groupby(customers.signup_date.dt.to_period("M")).size().reindex(cohort_counts.index)
retention = cohort_counts.div(cohort_size, axis=0) * 100
retention = retention.iloc[:, :13]
retention.to_csv(OUT / "cohort_retention.csv")

plt.figure(figsize=(14, 8))
sns.heatmap(retention.where(retention > 0), annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={"label": "% customers active"},
            linewidths=.4, annot_kws={"size": 7})
plt.title("Monthly Cohort Retention (% of cohort ordering in month N since signup)", fontweight="bold")
plt.xlabel("Months since signup"); plt.ylabel("Signup cohort")
plt.tight_layout(); plt.savefig(IMG / "05_cohort_retention.png", dpi=150); plt.close()
avg_ret = retention.mean()
insights["avg_retention_%"] = {f"M{k}": round(avg_ret[k], 1) for k in (1, 3, 6, 12) if k in avg_ret.index}

# ------------------------------------------------------------------ 7. CLV (historical + simple predictive)
cust_stats = valid_orders.groupby("customer_id").agg(orders=("order_id", "count"), revenue=("order_value", "sum"),
                                                     first=("order_date", "min"), last=("order_date", "max"))
cust_stats["tenure_months"] = ((cust_stats["last"] - cust_stats["first"]).dt.days / 30).clip(lower=1)
cust_stats["orders_per_month"] = cust_stats.orders / cust_stats.tenure_months
cust_stats["aov"] = cust_stats.revenue / cust_stats.orders
cust_stats = cust_stats.join(customers.set_index("customer_id")[["acquisition_channel", "city"]])
clv_channel = cust_stats.groupby("acquisition_channel").agg(avg_clv=("revenue", "mean"), avg_orders=("orders", "mean"),
                                                            customers=("revenue", "size")).sort_values("avg_clv", ascending=False)
clv_channel.to_csv(OUT / "clv_by_channel.csv")

# Simple predictive CLV: AOV x orders/month x assumed 12-month horizon x retention factor (Month-6 retention)
r6 = avg_ret.get(6, avg_ret.iloc[-1]) / 100
cust_stats["predicted_12m_clv"] = cust_stats.aov * cust_stats.orders_per_month * 12 * max(r6, 0.2)
cust_stats.to_csv(OUT / "customer_clv.csv")

plt.figure(figsize=(9, 5))
sns.barplot(x=clv_channel.avg_clv, y=clv_channel.index, palette="rocket")
plt.title("Average Customer Lifetime Value by Acquisition Channel (₹)", fontweight="bold"); plt.ylabel("")
plt.tight_layout(); plt.savefig(IMG / "06_clv_by_channel.png", dpi=150); plt.close()
insights["avg_clv"] = inr(cust_stats.revenue.mean())

# ------------------------------------------------------------------ 8. K-Means on RFM
X = StandardScaler().fit_transform(np.log1p(rfm[["Recency", "Frequency", "Monetary"]]))
inertia = [KMeans(k, n_init=10, random_state=42).fit(X).inertia_ for k in range(2, 9)]
km = KMeans(4, n_init=10, random_state=42).fit(X)
rfm["Cluster"] = km.labels_
cl = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(1)
cl["customers"] = rfm.Cluster.value_counts().sort_index()
cl.to_csv(OUT / "kmeans_clusters.csv")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(range(2, 9), inertia, marker="o", color=PALETTE[3]); ax[0].set_title("Elbow Method (K-Means)", fontweight="bold")
ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
sc = ax[1].scatter(rfm.Frequency, rfm.Monetary / 1e3, c=rfm.Cluster, cmap="viridis", s=10, alpha=.6)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_title("Customer Clusters (Frequency vs Monetary)", fontweight="bold")
ax[1].set_xlabel("Frequency (log)"); ax[1].set_ylabel("Monetary ₹K (log)")
plt.tight_layout(); plt.savefig(IMG / "07_kmeans_clusters.png", dpi=150); plt.close()

# ------------------------------------------------------------------ save
with open(OUT / "insights.json", "w", encoding="utf-8") as f:
    json.dump(insights, f, indent=2, ensure_ascii=False, default=str)
print(json.dumps(insights, indent=2, ensure_ascii=False, default=str))
print("\nCategory:\n", cat.round(1)); print("\nChannel:\n", channel.round(1)); print("\nSegments:\n", seg.round(1))
print("\nCluster:\n", cl); print("\nCLV by channel:\n", clv_channel.round(0))
print("\nPayment return %:\n", cod_return.round(1))
