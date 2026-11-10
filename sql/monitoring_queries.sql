-- Run against data/predictions.db (written by the API) for production monitoring.

-- Flag rate and latency per hour (alert on sudden jumps = data drift or attack)
SELECT substr(scored_at, 1, 13) AS hour_utc,
       COUNT(*)                 AS scored,
       SUM(is_fraud)            AS flagged,
       ROUND(100.0 * SUM(is_fraud) / COUNT(*), 2) AS flag_rate_pct,
       ROUND(AVG(latency_ms), 2) AS avg_latency_ms
FROM predictions
GROUP BY hour_utc
ORDER BY hour_utc DESC;

-- Highest-risk transactions awaiting analyst review
SELECT transaction_id, amount, fraud_probability, scored_at
FROM predictions
WHERE risk_level = 'high'
ORDER BY fraud_probability DESC
LIMIT 20;
