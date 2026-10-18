"""
Flask API Backend
Serves all building energy data endpoints and routes AI requests
to the IBM Granite model through WatsonX AI.
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import traceback
import os

from load_monitor import (
    generate_hourly_data,
    get_system_summary,
    get_total_load_series,
    compute_load_factor,
)
from peak_detector import (
    detect_peaks,
    get_peak_hours,
    get_demand_statistics,
)
from granite_ai import (
    get_load_balancing_recommendations,
    get_energy_optimization_insights,
    get_anomaly_explanation,
)

app = Flask(__name__)
CORS(app)

# ─── Serve Dashboard ──────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Serve the dashboard HTML directly so Edge opens it from http://localhost:5000"""
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
    return send_file(dashboard_path)


def _serialize(obj):
    """Convert pandas / numpy types to native Python for JSON serialization."""
    import numpy as np
    import pandas as pd
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def _df_to_records(df):
    return [{k: _serialize(v) for k, v in row.items()} for row in df.to_dict("records")]


# ─── Health Check ──────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "AI Smart Load Balancing Agent"})


# ─── Load Monitoring Data ──────────────────────────────────────────────────────

@app.route("/api/load-data", methods=["GET"])
def load_data():
    """Return hourly load data for all building systems."""
    try:
        hours = int(request.args.get("hours", 24))
        hours = max(1, min(hours, 168))   # clamp 1–168 h
        df = generate_hourly_data(hours=hours)
        total = get_total_load_series(df)
        summary = get_system_summary(df)
        load_factor = compute_load_factor(total)

        return jsonify({
            "success": True,
            "hours": hours,
            "load_factor": load_factor,
            "system_summary": _df_to_records(summary),
            "total_series": _df_to_records(total),
            "raw_data": _df_to_records(df),
        })
    except Exception:
        return jsonify({"success": False, "error": traceback.format_exc()}), 500


# ─── Peak Detection ────────────────────────────────────────────────────────────

@app.route("/api/peaks", methods=["GET"])
def peaks():
    """Detect and return peak load events."""
    try:
        hours = int(request.args.get("hours", 24))
        hours = max(1, min(hours, 168))
        z_threshold = float(request.args.get("z_threshold", 1.5))

        df = generate_hourly_data(hours=hours)
        total = get_total_load_series(df)
        demand_stats = get_demand_statistics(total)
        peak_events = detect_peaks(total, df, z_threshold=z_threshold)
        peak_hours  = get_peak_hours(total, top_n=5)

        peak_list = [
            {
                "timestamp": p.timestamp,
                "total_kw": p.total_kw,
                "z_score": p.z_score,
                "severity": p.severity,
                "systems_contributing": p.systems_contributing,
            }
            for p in peak_events
        ]

        return jsonify({
            "success": True,
            "demand_statistics": demand_stats,
            "peak_events": peak_list,
            "peak_hours": peak_hours,
            "total_peaks_detected": len(peak_list),
        })
    except Exception:
        return jsonify({"success": False, "error": traceback.format_exc()}), 500


# ─── AI Recommendations ────────────────────────────────────────────────────────

@app.route("/api/recommendations", methods=["GET"])
def recommendations():
    """Generate AI load balancing recommendations via IBM Granite."""
    try:
        hours = int(request.args.get("hours", 24))
        hours = max(1, min(hours, 168))

        df      = generate_hourly_data(hours=hours)
        total   = get_total_load_series(df)
        summary = get_system_summary(df)
        stats   = get_demand_statistics(total)
        peaks   = detect_peaks(total, df)

        peak_list = [
            {
                "timestamp": p.timestamp,
                "total_kw": p.total_kw,
                "z_score": p.z_score,
                "severity": p.severity,
                "systems_contributing": p.systems_contributing,
            }
            for p in peaks[:5]
        ]

        ai_text = get_load_balancing_recommendations(
            peak_events=peak_list,
            system_summary=_df_to_records(summary),
            demand_stats=stats,
        )

        return jsonify({
            "success": True,
            "ai_recommendations": ai_text,
            "context": {
                "peak_kw": stats["peak_kw"],
                "average_kw": stats["average_kw"],
                "peaks_analyzed": len(peak_list),
            },
        })
    except Exception:
        return jsonify({"success": False, "error": traceback.format_exc()}), 500


# ─── Energy Optimization Insights ─────────────────────────────────────────────

@app.route("/api/insights", methods=["GET"])
def insights():
    """Generate AI energy optimization insights via IBM Granite."""
    try:
        hours = int(request.args.get("hours", 24))
        hours = max(1, min(hours, 168))

        df          = generate_hourly_data(hours=hours)
        total       = get_total_load_series(df)
        summary     = get_system_summary(df)
        stats       = get_demand_statistics(total)
        load_factor = compute_load_factor(total)
        peak_hours  = get_peak_hours(total, top_n=3)

        ai_text = get_energy_optimization_insights(
            system_summary=_df_to_records(summary),
            demand_stats=stats,
            load_factor=load_factor,
            peak_hours=peak_hours,
        )

        return jsonify({
            "success": True,
            "ai_insights": ai_text,
            "load_factor": load_factor,
            "demand_statistics": stats,
        })
    except Exception:
        return jsonify({"success": False, "error": traceback.format_exc()}), 500


# ─── Anomaly Explanation ───────────────────────────────────────────────────────

@app.route("/api/explain-anomaly", methods=["POST"])
def explain_anomaly():
    """Ask IBM Granite to explain a specific anomaly event."""
    try:
        data = request.get_json(force=True)
        explanation = get_anomaly_explanation(data)
        return jsonify({"success": True, "explanation": explanation})
    except Exception:
        return jsonify({"success": False, "error": traceback.format_exc()}), 500


# ─── Full Dashboard Data (single call) ────────────────────────────────────────

@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    """
    Return all data needed by the dashboard in one call.
    AI text is excluded here for speed; fetch separately if needed.
    """
    try:
        hours = int(request.args.get("hours", 24))
        hours = max(1, min(hours, 168))

        df          = generate_hourly_data(hours=hours)
        total       = get_total_load_series(df)
        summary     = get_system_summary(df)
        stats       = get_demand_statistics(total)
        load_factor = compute_load_factor(total)
        peak_events = detect_peaks(total, df)
        peak_hours  = get_peak_hours(total, top_n=5)

        peak_list = [
            {
                "timestamp": p.timestamp,
                "total_kw": p.total_kw,
                "z_score": p.z_score,
                "severity": p.severity,
                "systems_contributing": p.systems_contributing,
            }
            for p in peak_events
        ]

        return jsonify({
            "success": True,
            "hours": hours,
            "load_factor": load_factor,
            "demand_statistics": stats,
            "system_summary": _df_to_records(summary),
            "total_series": _df_to_records(total),
            "peak_events": peak_list,
            "peak_hours": peak_hours,
            "total_peaks_detected": len(peak_list),
        })
    except Exception:
        return jsonify({"success": False, "error": traceback.format_exc()}), 500


if __name__ == "__main__":
    import io
    sys_stdout = __import__('sys').stdout
    try:
        __import__('sys').stdout = io.TextIOWrapper(sys_stdout.buffer, encoding='utf-8')
    except AttributeError:
        pass
    print("=" * 60)
    print("  AI Smart Load Balancing Agent - Starting Server")
    print("  http://localhost:5000")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=5000)
