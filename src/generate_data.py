"""
Synthetic E-Commerce data generator (India-focused).
Creates: customers, products, orders, order_items  -> data/*.csv
Run: python src/generate_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)
DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)

N_CUSTOMERS = 5000
START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")

# ---------------- Products ----------------
categories = {            # (min price, max price, margin)
    "Electronics":    (1500, 40000, 0.18),
    "Fashion":        (300, 5000, 0.35),
    "Home & Kitchen": (250, 8000, 0.22),
    "Beauty":         (150, 2500, 0.28),
    "Books":          (100, 900, 0.12),
    "Sports":         (400, 9000, 0.20),
    "Grocery":        (50, 1200, 0.08),
}
rows, pid = [], 1
for cat, (lo, hi, margin) in categories.items():
    for _ in range(40):
        price = round(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), -1)
        rows.append((pid, f"{cat[:3].upper()}-{pid:03d}", cat, price,
                     round(price * (1 - margin * rng.uniform(0.8, 1.2)), 2)))
        pid += 1
products = pd.DataFrame(rows, columns=["product_id", "product_name", "category", "price", "cost"])
prod_by_cat = {c: g.reset_index(drop=True) for c, g in products.groupby("category")}

# ---------------- Customers ----------------
cities = {"Mumbai":.14,"Delhi":.14,"Bengaluru":.12,"Hyderabad":.08,"Chennai":.07,"Kolkata":.06,
          "Pune":.07,"Ahmedabad":.05,"Lucknow":.05,"Jaipur":.05,"Chandigarh":.04,"Patna":.04,"Indore":.04,"Kochi":.05}
city_names = list(cities); city_p = np.array(list(cities.values())); city_p /= city_p.sum()
signup = START + pd.to_timedelta(rng.beta(1.3, 1.8, N_CUSTOMERS) * (END - START).days, unit="D")
customers = pd.DataFrame({
    "customer_id": np.arange(1, N_CUSTOMERS + 1),
    "signup_date": signup.normalize(),
    "city": rng.choice(city_names, N_CUSTOMERS, p=city_p),
    "gender": rng.choice(["Male", "Female", "Other"], N_CUSTOMERS, p=[.52, .46, .02]),
    "age": np.clip(rng.normal(32, 9, N_CUSTOMERS).astype(int), 18, 65),
    "acquisition_channel": rng.choice(["Organic Search","Paid Ads","Social Media","Referral","Email"],
                                      N_CUSTOMERS, p=[.30,.25,.22,.13,.10]),
})

# ---------------- Orders ----------------
# Each customer has a latent purchase rate + lifetime -> realistic heterogeneity for RFM / cohorts.
loyalty = rng.gamma(shape=1.2, scale=0.55, size=N_CUSTOMERS)   # orders per month
life_m  = rng.exponential(scale=9, size=N_CUSTOMERS)           # active lifetime (months)
festive = {10: 1.6, 11: 1.9, 12: 1.3, 1: 1.1}                  # Diwali / year-end sale effect

order_rows, item_rows, oid = [], [], 1
pay_methods = ["UPI", "Credit Card", "Debit Card", "COD", "Net Banking", "Wallet"]
pay_p = [.38, .17, .14, .20, .06, .05]
cat_list = list(categories)
cat_p = np.array([.20,.22,.14,.12,.08,.10,.14])

for c in customers.itertuples():
    i = c.customer_id - 1
    life_end = min(END, c.signup_date + pd.Timedelta(days=float(life_m[i]) * 30))
    cur = c.signup_date + pd.Timedelta(days=int(rng.integers(0, 5)))
    while cur <= life_end:
        n_items = int(np.clip(rng.poisson(0.8) + 1, 1, 6))
        cats = rng.choice(cat_list, n_items, p=cat_p)
        subtotal = 0.0
        for cat in cats:
            p = prod_by_cat[cat].iloc[int(rng.integers(0, 40))]
            qty = int(rng.choice([1, 2, 3], p=[.8, .15, .05]))
            disc = float(rng.choice([0, .05, .10, .20, .30], p=[.35, .2, .2, .15, .10]))
            item_rows.append((oid, int(p.product_id), qty, p.price, disc))
            subtotal += p.price * qty * (1 - disc)
        status = rng.choice(["Delivered", "Returned", "Cancelled"], p=[.88, .07, .05])
        order_rows.append((oid, c.customer_id, cur.normalize(), round(subtotal, 2),
                           rng.choice(pay_methods, p=pay_p), status))
        oid += 1
        gap = (rng.exponential(30 / max(loyalty[i], 0.05)) + 1) / festive.get(cur.month, 1.0)
        cur += pd.Timedelta(days=float(gap))

orders = pd.DataFrame(order_rows, columns=["order_id","customer_id","order_date","order_value","payment_method","order_status"])
items = pd.DataFrame(item_rows, columns=["order_id","product_id","quantity","unit_price","discount"])

for name, df in {"customers": customers, "products": products, "orders": orders, "order_items": items}.items():
    df.to_csv(DATA / f"{name}.csv", index=False)
    print(f"{name:12s} -> {len(df):>7,} rows")
