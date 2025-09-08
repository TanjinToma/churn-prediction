#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 10:10:12 2025

@author: tanjintoma
"""

import psycopg2
import pandas as pd

# Connection details
DB_CONFIG = {
    "host": "localhost",
    "dbname": "telco_db",
    "user": "postgres",
    "password": "Tt29041989@ma",
    "port": 5432
}

def load_csv_to_table(csv_path, table_name, conn):
    df = pd.read_csv(csv_path)
    cursor = conn.cursor()

    for _, row in df.iterrows():
        cols = ",".join(df.columns)
        vals = [row[c] for c in df.columns]
        placeholders = ",".join(["%s"] * len(vals))
        sql = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders}) ON CONFLICT DO NOTHING;"
        cursor.execute(sql, vals)

    conn.commit()
    cursor.close()
    print(f"✅ Loaded {len(df)} rows into {table_name}")

def main():
    conn = psycopg2.connect(**DB_CONFIG)

    load_csv_to_table("../data/customers.csv", "customers", conn)
    load_csv_to_table("../data/contracts.csv", "contracts", conn)
    load_csv_to_table("../data/billing.csv", "billing", conn)
    load_csv_to_table("../data/services.csv", "services", conn)
    load_csv_to_table("../data/churn.csv", "churn", conn)
    load_csv_to_table("../data/telco_sequential.csv", "telco_sequential", conn)

    conn.close()

if __name__ == "__main__":
    main()
