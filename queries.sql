-- SQL used in the notebook (run on an in-memory SQLite table named `customers`,
-- one row per customer, columns as in data/hillstrom.csv)

-- 1. Headline metrics per experiment group
SELECT segment,
       COUNT(*)                                          AS customers,
       ROUND(AVG(visit) * 100, 2)                        AS visit_rate_pct,
       ROUND(AVG(conversion) * 100, 3)                   AS conversion_rate_pct,
       ROUND(AVG(spend), 3)                              AS spend_per_customer,
       ROUND(SUM(spend), 0)                              AS total_spend,
       ROUND(SUM(spend) / NULLIF(SUM(conversion), 0), 2) AS spend_per_buyer
FROM customers
GROUP BY segment
ORDER BY spend_per_customer DESC;

-- 2. Visit rate by past buyer type and experiment group
SELECT CASE WHEN mens = 1 AND womens = 1 THEN 'Bought both'
            WHEN mens = 1 THEN 'Bought mens only'
            ELSE 'Bought womens only' END               AS buyer_type,
       segment,
       COUNT(*)                                          AS customers,
       ROUND(AVG(visit) * 100, 2)                        AS visit_rate_pct
FROM customers
GROUP BY buyer_type, segment
ORDER BY buyer_type, segment;
