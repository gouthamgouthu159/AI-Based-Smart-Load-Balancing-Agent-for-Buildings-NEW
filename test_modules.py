#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick validation script – tests all modules without starting the server.
Run: python test_modules.py
"""
import sys
import io

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 60)
print("  AI Smart Load Balancing Agent - Module Validation")
print("=" * 60)

# -- 1. Load Monitor ----------------------------------------------------------
print("\n[1/4] Testing Load Monitor...")
try:
    from load_monitor import (
        generate_hourly_data, get_system_summary,
        get_total_load_series, compute_load_factor,
    )
    df = generate_hourly_data(hours=24, seed=42)
    total   = get_total_load_series(df)
    summary = get_system_summary(df)
    lf      = compute_load_factor(total)
    assert len(df) > 0, "Empty dataframe"
    assert "load_kw" in df.columns
    print(f"    [OK] Generated {len(df)} rows | Load factor: {lf}")
    print(f"    Systems: {df['system'].unique().tolist()}")
except Exception as e:
    print(f"    [FAIL] {e}")
    sys.exit(1)

# -- 2. Peak Detector ---------------------------------------------------------
print("\n[2/4] Testing Peak Detector...")
try:
    from peak_detector import detect_peaks, get_peak_hours, get_demand_statistics
    peaks      = detect_peaks(total, df)
    peak_hours = get_peak_hours(total, top_n=5)
    stats      = get_demand_statistics(total)
    print(f"    [OK] Detected {len(peaks)} peak events")
    print(f"    Peak demand: {stats['peak_kw']} kW | Avg: {stats['average_kw']} kW")
    print(f"    PAR: {stats['peak_to_avg_ratio']} | Top peak hour: {peak_hours[0]['hour']}:00")
except Exception as e:
    print(f"    [FAIL] {e}")
    sys.exit(1)

# -- 3. IBM Granite Import -----------------------------------------------------
print("\n[3/4] Testing IBM Granite module import...")
try:
    from granite_ai import IBM_API_KEY, IBM_PROJECT_ID, IBM_WX_URL
    assert IBM_API_KEY and IBM_PROJECT_ID, "Missing credentials"
    print(f"    [OK] Credentials loaded")
    print(f"    API Key : {IBM_API_KEY[:8]}...")
    print(f"    Project : {IBM_PROJECT_ID}")
    print(f"    Endpoint: {IBM_WX_URL}")
except Exception as e:
    print(f"    [FAIL] {e}")
    sys.exit(1)

# -- 4. Flask App Import -------------------------------------------------------
print("\n[4/4] Testing Flask app import...")
try:
    import app as flask_app
    client = flask_app.app.test_client()
    resp   = client.get("/api/health")
    assert resp.status_code == 200, f"Status {resp.status_code}"
    payload = resp.get_json()
    assert payload["status"] == "ok"
    print(f"    [OK] Health check passed: {payload}")

    resp2 = client.get("/api/dashboard?hours=24")
    assert resp2.status_code == 200
    d = resp2.get_json()
    assert d["success"]
    print(f"    [OK] Dashboard endpoint: peaks={d['total_peaks_detected']}, LF={d['load_factor']}")
except Exception as e:
    print(f"    [FAIL] {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("  ALL MODULES VALIDATED SUCCESSFULLY!")
print("  Run  python app.py  to start the server.")
print("  Then open  dashboard.html  in your browser.")
print("=" * 60)
