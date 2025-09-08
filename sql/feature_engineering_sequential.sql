-- Feature engineering for sequential churn prediction
-- Creates sequences of features per customer, with final churn label

WITH base AS (
    SELECT
        customer_id,
        month,
        monthly_charges,
        data_usage_gb,
        complaints,
        churn
    FROM telco_sequential
    ORDER BY customer_id, month
),
agg AS (
    SELECT
        customer_id,
        array_agg(monthly_charges ORDER BY month) AS seq_monthly_charges,
        array_agg(data_usage_gb ORDER BY month)   AS seq_data_usage,
        array_agg(complaints ORDER BY month)      AS seq_complaints,
        MAX(month) AS last_month
    FROM base
    GROUP BY customer_id
)

-- Attach churn label = churn at the last month
SELECT
    a.customer_id,
    a.seq_monthly_charges,
    a.seq_data_usage,
    a.seq_complaints,
    b.churn AS churn_label
FROM agg a
JOIN base b
    ON a.customer_id = b.customer_id AND a.last_month = b.month
ORDER BY a.customer_id;
