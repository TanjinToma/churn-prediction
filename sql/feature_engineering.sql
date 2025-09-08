WITH base AS (
    SELECT
        c.customer_id,
        c.gender,
        c.senior_citizen,
        c.partner,
        c.dependents,
        ct.contract_type,
        ct.paperless_billing,
        ct.payment_method,
        b.tenure,
        b.monthly_charges,
        b.total_charges,
        s.phone_service,
        s.multiple_lines,
        s.internet_service,
        s.online_security,
        s.online_backup,
        s.device_protection,
        s.tech_support,
        s.streaming_tv,
        s.streaming_movies,
        ch.churn
    FROM customers c
    LEFT JOIN contracts ct ON c.customer_id = ct.customer_id
    LEFT JOIN billing b ON c.customer_id = b.customer_id
    LEFT JOIN services s ON c.customer_id = s.customer_id
    LEFT JOIN churn ch ON c.customer_id = ch.customer_id
)
SELECT * FROM base;
