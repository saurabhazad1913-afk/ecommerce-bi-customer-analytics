-- =====================================================================
-- E-Commerce BI: SQL queries (PostgreSQL / MySQL 8+ compatible)
-- Tables: customers, products, orders, order_items
-- Load the CSVs from /data into your database first.
-- =====================================================================

-- 1. Core KPIs (delivered orders only)
SELECT
    ROUND(SUM(order_value), 0)                     AS total_revenue,
    COUNT(*)                                       AS delivered_orders,
    COUNT(DISTINCT customer_id)                    AS active_customers,
    ROUND(SUM(order_value) / COUNT(*), 0)          AS avg_order_value
FROM orders
WHERE order_status = 'Delivered';

-- 2. Monthly revenue with Month-over-Month growth
WITH monthly AS (
    SELECT DATE_TRUNC('month', order_date) AS month,
           SUM(order_value)                AS revenue
    FROM orders
    WHERE order_status = 'Delivered'
    GROUP BY 1
)
SELECT month, revenue,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
             / NULLIF(LAG(revenue) OVER (ORDER BY month), 0), 1) AS mom_growth_pct
FROM monthly
ORDER BY month;

-- 3. Category revenue, profit and margin
SELECT p.category,
       ROUND(SUM(oi.unit_price * oi.quantity * (1 - oi.discount)), 0)                      AS revenue,
       ROUND(SUM(oi.unit_price * oi.quantity * (1 - oi.discount) - p.cost * oi.quantity), 0) AS profit,
       ROUND(100.0 * SUM(oi.unit_price * oi.quantity * (1 - oi.discount) - p.cost * oi.quantity)
             / SUM(oi.unit_price * oi.quantity * (1 - oi.discount)), 1)                    AS margin_pct
FROM order_items oi
JOIN products p ON p.product_id = oi.product_id
JOIN orders   o ON o.order_id   = oi.order_id AND o.order_status = 'Delivered'
GROUP BY p.category
ORDER BY revenue DESC;

-- 4. Top 10 customers by lifetime revenue
SELECT customer_id, COUNT(*) AS orders, ROUND(SUM(order_value), 0) AS lifetime_revenue
FROM orders
WHERE order_status = 'Delivered'
GROUP BY customer_id
ORDER BY lifetime_revenue DESC
LIMIT 10;

-- 5. RFM scores (NTILE-based) -- snapshot = day after last order
WITH snap AS (SELECT MAX(order_date) + INTERVAL '1 day' AS d FROM orders),
rfm AS (
    SELECT o.customer_id,
           (SELECT d FROM snap)::date - MAX(o.order_date)::date AS recency,
           COUNT(*)                                             AS frequency,
           SUM(o.order_value)                                   AS monetary
    FROM orders o
    WHERE o.order_status = 'Delivered'
    GROUP BY o.customer_id
)
SELECT customer_id, recency, frequency, monetary,
       NTILE(5) OVER (ORDER BY recency DESC)  AS r_score,
       NTILE(5) OVER (ORDER BY frequency)     AS f_score,
       NTILE(5) OVER (ORDER BY monetary)      AS m_score
FROM rfm;

-- 6. Cohort retention (signup month vs months since signup)
WITH cohort AS (
    SELECT c.customer_id, DATE_TRUNC('month', c.signup_date) AS cohort_month
    FROM customers c
),
activity AS (
    SELECT DISTINCT o.customer_id, DATE_TRUNC('month', o.order_date) AS active_month
    FROM orders o
    WHERE o.order_status = 'Delivered'
)
SELECT ch.cohort_month,
       (EXTRACT(YEAR FROM a.active_month) - EXTRACT(YEAR FROM ch.cohort_month)) * 12
       + (EXTRACT(MONTH FROM a.active_month) - EXTRACT(MONTH FROM ch.cohort_month)) AS month_number,
       COUNT(DISTINCT a.customer_id) AS active_customers
FROM cohort ch
JOIN activity a ON a.customer_id = ch.customer_id
GROUP BY 1, 2
ORDER BY 1, 2;

-- 7. Repeat purchase rate
SELECT ROUND(100.0 * SUM(CASE WHEN n > 1 THEN 1 ELSE 0 END) / COUNT(*), 1) AS repeat_rate_pct
FROM (SELECT customer_id, COUNT(*) AS n FROM orders
      WHERE order_status = 'Delivered' GROUP BY customer_id) t;

-- 8. Revenue & CLV by acquisition channel
SELECT c.acquisition_channel,
       COUNT(DISTINCT c.customer_id)                              AS customers,
       ROUND(SUM(o.order_value) / COUNT(DISTINCT c.customer_id), 0) AS revenue_per_customer
FROM customers c
JOIN orders o ON o.customer_id = c.customer_id AND o.order_status = 'Delivered'
GROUP BY c.acquisition_channel
ORDER BY revenue_per_customer DESC;

-- 9. City-wise revenue ranking
SELECT c.city, ROUND(SUM(o.order_value), 0) AS revenue,
       RANK() OVER (ORDER BY SUM(o.order_value) DESC) AS city_rank
FROM orders o JOIN customers c USING (customer_id)
WHERE o.order_status = 'Delivered'
GROUP BY c.city;

-- 10. Pareto: share of revenue from top 20% customers
WITH cust AS (
    SELECT customer_id, SUM(order_value) AS rev,
           NTILE(5) OVER (ORDER BY SUM(order_value) DESC) AS quintile
    FROM orders WHERE order_status = 'Delivered' GROUP BY customer_id
)
SELECT ROUND(100.0 * SUM(CASE WHEN quintile = 1 THEN rev END) / SUM(rev), 1) AS top20_revenue_share_pct
FROM cust;
