-- Customers table
CREATE TABLE customers (
    customer_id VARCHAR PRIMARY KEY,
    gender VARCHAR,
    senior_citizen INT,
    partner VARCHAR,
    dependents VARCHAR
);

-- Contracts table
CREATE TABLE contracts (
    contract_id INT PRIMARY KEY,
    customer_id VARCHAR REFERENCES customers(customer_id),
    contract_type VARCHAR,
    paperless_billing VARCHAR,
    payment_method VARCHAR
);

-- Billing table
CREATE TABLE billing (
    billing_id INT PRIMARY KEY,
    customer_id VARCHAR REFERENCES customers(customer_id),
    tenure INT,
    monthly_charges NUMERIC,
    total_charges NUMERIC
);

-- Services table
CREATE TABLE services (
    service_id INT PRIMARY KEY,
    customer_id VARCHAR REFERENCES customers(customer_id),
    phone_service VARCHAR,
    multiple_lines VARCHAR,
    internet_service VARCHAR,
    online_security VARCHAR,
    online_backup VARCHAR,
    device_protection VARCHAR,
    tech_support VARCHAR,
    streaming_tv VARCHAR,
    streaming_movies VARCHAR
);

-- Churn table
CREATE TABLE churn (
    churn_id INT PRIMARY KEY,
    customer_id VARCHAR REFERENCES customers(customer_id),
    churn VARCHAR CHECK (churn IN ('Yes', 'No'))
);


-- Sequential churn table
CREATE TABLE telco_sequential (
    customer_id VARCHAR REFERENCES customers(customer_id),
    month INT,
    monthly_charges DOUBLE PRECISION,
    data_usage_gb DOUBLE PRECISION,
    complaints INT,
    churn INT CHECK (churn IN (0, 1)),
    PRIMARY KEY (customer_id, month)
);