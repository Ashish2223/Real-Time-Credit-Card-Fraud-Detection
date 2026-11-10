-- name: class_distribution
SELECT Class,
       COUNT(*) AS n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 4) AS pct
FROM transactions
GROUP BY Class;

-- name: fraud_rate_by_hour_of_day
SELECT CAST(Time / 3600 AS INTEGER) % 24 AS hour,
       COUNT(*)                          AS txns,
       SUM(Class)                        AS frauds,
       ROUND(100.0 * SUM(Class) / COUNT(*), 3) AS fraud_rate_pct
FROM transactions
GROUP BY hour
ORDER BY hour;

-- name: fraud_rate_by_amount_bucket
SELECT CASE WHEN Amount < 5    THEN '1) < 5'
            WHEN Amount < 20   THEN '2) 5-20'
            WHEN Amount < 100  THEN '3) 20-100'
            WHEN Amount < 500  THEN '4) 100-500'
            ELSE                    '5) 500+' END AS amount_bucket,
       COUNT(*)   AS txns,
       SUM(Class) AS frauds,
       ROUND(100.0 * SUM(Class) / COUNT(*), 3) AS fraud_rate_pct
FROM transactions
GROUP BY amount_bucket
ORDER BY amount_bucket;

-- name: amount_stats_by_class
SELECT Class,
       ROUND(AVG(Amount), 2) AS avg_amount,
       ROUND(MIN(Amount), 2) AS min_amount,
       ROUND(MAX(Amount), 2) AS max_amount
FROM transactions
GROUP BY Class;

-- name: fraud_rate_by_amount_decile
WITH ranked AS (
    SELECT Amount, Class, NTILE(10) OVER (ORDER BY Amount) AS decile FROM transactions
)
SELECT decile,
       ROUND(MIN(Amount), 2) AS from_amount,
       ROUND(MAX(Amount), 2) AS to_amount,
       SUM(Class)            AS frauds,
       ROUND(100.0 * SUM(Class) / COUNT(*), 3) AS fraud_rate_pct
FROM ranked
GROUP BY decile
ORDER BY decile;

-- name: rolling_6h_fraud_rate
WITH hourly AS (
    SELECT CAST(Time / 3600 AS INTEGER) AS hour_idx, COUNT(*) AS txns, SUM(Class) AS frauds
    FROM transactions
    GROUP BY hour_idx
)
SELECT hour_idx, txns, frauds,
       ROUND(100.0 * SUM(frauds) OVER w / SUM(txns) OVER w, 3) AS rolling_6h_fraud_rate_pct
FROM hourly
WINDOW w AS (ORDER BY hour_idx ROWS BETWEEN 5 PRECEDING AND CURRENT ROW)
ORDER BY hour_idx
LIMIT 12;
