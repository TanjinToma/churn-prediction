#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 15:21:51 2025

@author: tanjintoma
"""

import psycopg2
import pandas as pd

def get_connection():    
    # Connection parameters
    conn = psycopg2.connect(
        dbname="telco_db",     # telco_db (schema name)
        user="postgres",       # replace with your user
        password="xxxxxxxx",   # replace with your password
        host="localhost",
        port="5432"
    )
    return conn



def load_features(sql_file: str) -> pd.DataFrame:
    conn = get_connection()
    with open(sql_file, "r") as f:
        query = f.read()
    df = pd.read_sql(query, conn)
    conn.close()
    return df
