"""PostgreSQL persistence for mastitis observation records.

Stores every record submitted via /append_cow_record into
bovin_mastitis.mastitis_observations.
"""

import os
from datetime import datetime, timezone
from functools import lru_cache

import psycopg
from psycopg.rows import dict_row
from fastapi import HTTPException

# Fallback default; the DATABASE_URL environment variable always wins.
# External Render hostname: resolves from anywhere (local dev, CI, Render).
DEFAULT_DATABASE_URL = (
    "postgresql://bovine:Cs15Ox4Qb2YU4vOIlcgfeqmosEv5ljxH"
    "@dpg-dask9fo473hc738mp5tg-a.singapore-postgres.render.com/bovine_mastitis"
)

SCHEMA = "bovin_mastitis"
TABLE = f"{SCHEMA}.mastitis_observations"


def get_connection_string() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


@lru_cache(maxsize=1)
def _connect() -> psycopg.Connection:
    conn = psycopg.connect(
        get_connection_string(), autocommit=True, row_factory=dict_row
    )
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    return conn


def get_connection() -> psycopg.Connection:
    """Return a cached, autocommit connection, reconnecting if dead."""
    try:
        conn = _connect()
        conn.execute("SELECT 1")
        return conn
    except (psycopg.OperationalError, psycopg.InterfaceError):
        _connect.cache_clear()
        conn = _connect()
        conn.execute("SELECT 1")
        return conn


def _safe_float(value):
    return None if value is None else float(value)


def insert_observation(inp) -> int:
    """Insert one observation row; returns the generated row id."""
    conn = get_connection()
    row = conn.execute(
        f"""
        INSERT INTO {TABLE} (
            farmer_id, cow_id, timestamp,
            milk_ec, milk_temperature_c, udder_temperature_c,
            activity_index, milk_colour_code,
            milk_ec_baseline, milk_temp_baseline_c, udder_temp_baseline_c,
            activity_baseline,
            milk_ec_deviation, milk_temp_deviation_c, udder_temp_deviation_c,
            activity_deviation,
            milk_ec_slope, udder_temp_slope, activity_slope,
            milk_ec_variability, udder_temp_variability, activity_variability,
            risk_score, risk_label, forecast_horizon_days
        ) VALUES (
            %s, %s, %s,
            %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s,
            %s, %s, %s,
            %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s
        )
        RETURNING id
        """,
        (
            inp.farmer_id,
            inp.cow_id,
            datetime.now(timezone.utc),
            inp.milk_ec, inp.milk_temperature_c, inp.udder_temperature_c,
            inp.activity_index, inp.milk_colour_code,
            inp.milk_ec_baseline, inp.milk_temp_baseline_c, inp.udder_temp_baseline_c,
            inp.activity_baseline,
            inp.milk_ec - inp.milk_ec_baseline,
            inp.milk_temperature_c - inp.milk_temp_baseline_c,
            inp.udder_temperature_c - inp.udder_temp_baseline_c,
            inp.activity_index - inp.activity_baseline,
            0.0, 0.0, 0.0,          # slopes: unknown for single-shot input
            0.0, 0.0, 0.0,          # variability: unknown for single-shot input
            getattr(inp, "risk_score", None),
            getattr(inp, "risk_label", None),
            inp.forecast_horizon_days,
        ),
    )
    return int(row.fetchone()["id"])


def insert_observation_with_prediction(inp, prediction: dict) -> int:
    """Insert one observation row together with its risk prediction."""
    conn = get_connection()
    row = conn.execute(
        f"""
        INSERT INTO {TABLE} (
            farmer_id, cow_id, timestamp,
            milk_ec, milk_temperature_c, udder_temperature_c,
            activity_index, milk_colour_code,
            milk_ec_baseline, milk_temp_baseline_c, udder_temp_baseline_c,
            activity_baseline,
            milk_ec_deviation, milk_temp_deviation_c, udder_temp_deviation_c,
            activity_deviation,
            milk_ec_slope, udder_temp_slope, activity_slope,
            milk_ec_variability, udder_temp_variability, activity_variability,
            risk_score, risk_label, forecast_horizon_days
        ) VALUES (
            %s, %s, %s,
            %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s,
            %s, %s, %s,
            %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s
        )
        RETURNING id
        """,
        (
            inp.farmer_id,
            inp.cow_id,
            datetime.now(timezone.utc),
            inp.milk_ec, inp.milk_temperature_c, inp.udder_temperature_c,
            inp.activity_index, inp.milk_color_code if hasattr(inp, "milk_color_code") else inp.milk_colour_code,
            inp.milk_ec_baseline, inp.milk_temp_baseline_c, inp.udder_temp_baseline_c,
            inp.activity_baseline,
            inp.milk_ec - inp.milk_ec_baseline,
            inp.milk_temperature_c - inp.milk_temp_baseline_c,
            inp.udder_temperature_c - inp.udder_temp_baseline_c,
            inp.activity_index - inp.activity_baseline,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            prediction.get("mastitis_risk_probability"),
            prediction.get("risk_category"),
            inp.forecast_horizon_days,
        ),
    )
    return int(row.fetchone()["id"])


def fetch_latest_record_for_cow(cow_id: str) -> dict | None:
    """Return the most recent observation for a cow (any farmer), or None."""
    conn = get_connection()
    row = conn.execute(
        f"""
        SELECT * FROM {TABLE}
        WHERE LOWER(cow_id) = LOWER(%s)
        ORDER BY timestamp DESC, id DESC
        LIMIT 1
        """,
        (cow_id.strip(),),
    ).fetchone()
    return dict(row) if row else None


def fetch_latest_record_per_cow() -> list[dict]:
    """Return the most recent observation for every distinct cow."""
    conn = get_connection()
    rows = conn.execute(
        f"""
        SELECT DISTINCT ON (LOWER(cow_id)) *
        FROM {TABLE}
        ORDER BY LOWER(cow_id), timestamp DESC, id DESC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def record_to_cow_sensor_input_dict(record: dict) -> dict:
    """Map a DB row into the CowSensorInput field naming used by the API."""
    # int(float(...)) tolerates rows stored as '3.0', 3.0, or 3.
    colour = record.get("milk_colour_code")
    horizon = record.get("forecast_horizon_days")
    return {
        "farmer_id": record.get("farmer_id"),
        "cow_id": record.get("cow_id"),
        "milk_ec": _safe_float(record.get("milk_ec")),
        "milk_temperature_c": _safe_float(record.get("milk_temperature_c")),
        "udder_temperature_c": _safe_float(record.get("udder_temperature_c")),
        "activity_index": _safe_float(record.get("activity_index")),
        "milk_colour_code": int(float(colour or 0)),
        "milk_ec_baseline": _safe_float(record.get("milk_ec_baseline")),
        "milk_temp_baseline_c": _safe_float(record.get("milk_temp_baseline_c")),
        "udder_temp_baseline_c": _safe_float(record.get("udder_temp_baseline_c")),
        "activity_baseline": _safe_float(record.get("activity_baseline")),
        "forecast_horizon_days": int(float(horizon or 7)),
    }
