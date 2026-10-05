# stream_data.py
import time
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


def _ensure_table_exists(connection, table_name: str):
    create_table_query = f"""CREATE TABLE IF NOT EXISTS {table_name} (
    trait VARCHAR(255),
    axis_1 DOUBLE PRECISION,
    axis_2 DOUBLE PRECISION,
    axis_3 DOUBLE PRECISION,
    axis_4 DOUBLE PRECISION,
    axis_5 DOUBLE PRECISION,
    axis_6 DOUBLE PRECISION,
    axis_7 DOUBLE PRECISION,
    axis_8 DOUBLE PRECISION,
    axis_9 DOUBLE PRECISION,
    axis_10 DOUBLE PRECISION,
    axis_11 DOUBLE PRECISION,
    axis_12 DOUBLE PRECISION,
    axis_13 DOUBLE PRECISION,
    axis_14 DOUBLE PRECISION,
    time TIMESTAMP
)"""
    with connection.cursor() as cursor:
        cursor.execute(create_table_query)
    connection.commit()


def read_recent_data(db_url: str, table_name: str) -> pd.DataFrame:
    query = f"""SELECT time, axis_1, axis_2, axis_3, axis_4,
        axis_5, axis_6, axis_7, axis_8
        FROM {table_name}
        WHERE time >= (
            SELECT MAX(time) FROM {table_name}
        ) - INTERVAL '90 seconds'
        ORDER BY time"""
    with psycopg2.connect(db_url) as connection:
        _ensure_table_exists(connection, table_name)
        return pd.read_sql_query(query, connection)

def stream_DF_to_neon(axis_data: pd.DataFrame, table_name: str, delay_seconds: float, db_url: str):
    """
    Streams dataframe rows one by one into a Neon database every `delay_seconds`.
    """
    # connect to the Neon postgreSQL database
    connection = psycopg2.connect(db_url)
    _ensure_table_exists(connection, table_name)
    cursor = connection.cursor()


    print("data stream receiving every 0.1 seconds")
    insert_query = f"""INSERT INTO {table_name} (Trait, Axis_1, Axis_2, Axis_3, Axis_4, Axis_5, Axis_6, Axis_7, Axis_8, Axis_9, Axis_10, Axis_11, Axis_12, Axis_13, Axis_14, Time) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s)"""
    try:
        for index, row in axis_data.iterrows():
            values = tuple(None if pd.isna(value) else value for value in row)
            cursor.execute(insert_query, values)
            connection.commit()
            print(f"[{time.strftime('%H:%M:%S')}] Transmitted Row {index + 1}/{len(axis_data)}")
            time.sleep(delay_seconds)

        print("Data uploaded successfully to Neon!")

    except KeyboardInterrupt:
        print("\nStreaming paused by user.")
    finally:
        cursor.close()
        connection.close()

def ensure_lab_tables(connection):
    """
    Create the three tables required by the lab:
      - training_data       : the RMBR4-2_export_test.csv training set
      - lab_streaming_data  : the synthetic test stream
      - anomaly_events      : detected Alert/Error events
    """
    with connection.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_data (
                time TIMESTAMP,
                axis_1 DOUBLE PRECISION,
                axis_2 DOUBLE PRECISION,
                axis_3 DOUBLE PRECISION,
                axis_4 DOUBLE PRECISION,
                axis_5 DOUBLE PRECISION,
                axis_6 DOUBLE PRECISION,
                axis_7 DOUBLE PRECISION,
                axis_8 DOUBLE PRECISION
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS lab_streaming_data (
                time TIMESTAMP,
                axis_2 DOUBLE PRECISION,
                axis_2_prediction DOUBLE PRECISION,
                axis_2_residual DOUBLE PRECISION,
                segment VARCHAR(32)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_events (
                axis VARCHAR(64),
                event_type VARCHAR(16),
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration_seconds DOUBLE PRECISION,
                max_deviation DOUBLE PRECISION
            )
        """)
    connection.commit()


def upload_training_data(df, db_url, table_name="training_data"):
    """Bulk-insert training data into Neon using execute_values."""
    conn = psycopg2.connect(db_url)
    try:
        ensure_lab_tables(conn)
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE {table_name}")
            rows = [
                (
                    row["Time"],
                    row["Axis #1"], row["Axis #2"],
                    row["Axis #3"], row["Axis #4"],
                    row["Axis #5"], row["Axis #6"],
                    row["Axis #7"], row["Axis #8"],
                )
                for _, row in df.iterrows()
            ]
            execute_values(
                cur,
                f"INSERT INTO {table_name} "
                "(time, axis_1, axis_2, axis_3, axis_4, "
                " axis_5, axis_6, axis_7, axis_8) VALUES %s",
                rows,
                page_size=500,
            )
        conn.commit()
    finally:
        conn.close()


def fetch_training_data(db_url, table_name="training_data"):
    """Pull training data back from Neon into a DataFrame."""
    with psycopg2.connect(db_url) as conn:
        query = f"SELECT * FROM {table_name} ORDER BY time"
        return pd.read_sql_query(query, conn)


def upload_synthetic_stream(df, db_url, table_name="lab_streaming_data"):
    """Bulk-insert synthetic stream into Neon using execute_values."""
    conn = psycopg2.connect(db_url)
    try:
        ensure_lab_tables(conn)
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE {table_name}")
            rows = [
                (
                    row["Time"],
                    row["Axis #2"],
                    row["Axis #2_prediction"],
                    row["Axis #2_residual"],
                    row["segment"],
                )
                for _, row in df.iterrows()
            ]
            execute_values(
                cur,
                f"INSERT INTO {table_name} "
                "(time, axis_2, axis_2_prediction, "
                " axis_2_residual, segment) VALUES %s",
                rows,
                page_size=500,
            )
        conn.commit()
    finally:
        conn.close()


def fetch_synthetic_stream(db_url, table_name="lab_streaming_data"):
    """Pull the synthetic test stream back from Neon, ordered by time."""
    with psycopg2.connect(db_url) as conn:
        query = (
            f"SELECT time, axis_2, axis_2_prediction, "
            f"axis_2_residual, segment "
            f"FROM {table_name} ORDER BY time"
        )
        return pd.read_sql_query(query, conn)


def log_anomaly_events(events, db_url, table_name="anomaly_events"):
    """Insert detected events into Neon (truncate-and-insert)."""
    conn = psycopg2.connect(db_url)
    try:
        ensure_lab_tables(conn)
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE {table_name}")
            insert_sql = (
                f"INSERT INTO {table_name} "
                "(axis, event_type, start_time, end_time, "
                " duration_seconds, max_deviation) "
                "VALUES (%s,%s,%s,%s,%s,%s)"
            )
            for e in events:
                cur.execute(insert_sql, (
                    e["axis"],
                    e["type"],
                    e["start"],
                    e["end"],
                    e["duration_seconds"],
                    e["max_deviation"],
                ))
        conn.commit()
    finally:
        conn.close()


def fetch_anomaly_events(db_url, table_name="anomaly_events"):
    """Read logged events back from Neon."""
    with psycopg2.connect(db_url) as conn:
        return pd.read_sql_query(
            f"SELECT * FROM {table_name} ORDER BY start_time", conn
        )
