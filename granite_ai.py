"""
IBM Granite AI Integration Module
Connects to IBM WatsonX AI (IBM Granite model) to generate intelligent
load balancing recommendations and energy optimization insights.
"""

import os
from dotenv import load_dotenv

# Load from config.env (our renamed .env file)
load_dotenv("config.env")

IBM_API_KEY    = os.getenv("IBM_API_KEY", "b819c7c4-a174-4f2c-9a23-68e0eefe3d48")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID", "faf682b2-e591-4a06-9f2c-2602ecb70aa8")
IBM_WX_URL     = os.getenv("IBM_WX_URL", "https://us-south.ml.cloud.ibm.com")

_client = None


def _get_client():
    """Lazily initialize the WatsonX AI client."""
    global _client
    if _client is None:
        from ibm_watsonx_ai import APIClient, Credentials
        credentials = Credentials(url=IBM_WX_URL, api_key=IBM_API_KEY)
        _client = APIClient(credentials)
    return _client


def _call_granite(prompt: str, max_tokens: int = 900) -> str:
    """
    Call IBM Granite via WatsonX AI and return the generated text.
    Falls back to a descriptive error string if the call fails.
    """
    try:
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Params

        model = ModelInference(
            model_id="ibm/granite-13b-instruct-v2",
            api_client=_get_client(),
            project_id=IBM_PROJECT_ID,
            params={
                Params.MAX_NEW_TOKENS:   max_tokens,
                Params.MIN_NEW_TOKENS:   60,
                Params.TEMPERATURE:      0.7,
                Params.TOP_P:            0.9,
                Params.REPETITION_PENALTY: 1.1,
            },
        )
        response = model.generate_text(prompt=prompt)
        return response.strip()

    except Exception as exc:
        return f"[Granite unavailable – {exc}]"


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def get_load_balancing_recommendations(
    peak_events: list,
    system_summary: list,
    demand_stats: dict,
) -> str:
    """
    Ask IBM Granite to generate smart load balancing recommendations.
    """
    # Build a concise context string
    peak_desc = "\n".join(
        f"  • {p['timestamp']} – {p['total_kw']} kW (severity: {p['severity']}, "
        f"top systems: {', '.join(p.get('systems_contributing', []))})"
        for p in peak_events[:5]
    ) or "  • No significant peaks detected."

    sys_desc = "\n".join(
        f"  • {s['system']}: avg {s['avg_kw']} kW, peak {s['max_kw']} kW"
        for s in system_summary
    )

    prompt = f"""You are an AI energy management expert for commercial buildings.
Analyze the following building electricity data and provide 5 specific, actionable load balancing recommendations.

=== BUILDING ENERGY SUMMARY ===
Peak Demand: {demand_stats['peak_kw']} kW
Average Demand: {demand_stats['average_kw']} kW
Peak-to-Average Ratio: {demand_stats['peak_to_avg_ratio']}
Load Factor: {round(demand_stats['average_kw'] / demand_stats['peak_kw'], 2) if demand_stats['peak_kw'] > 0 else 'N/A'}

=== TOP PEAK EVENTS ===
{peak_desc}

=== SYSTEM CONSUMPTION ===
{sys_desc}

=== TASK ===
Provide exactly 5 numbered, specific load-shifting and balancing recommendations that:
1. Target the identified peak periods
2. Address the highest-consuming systems
3. Include specific time windows for load shifting
4. Are practical for building operators to implement immediately

Recommendations:"""

    return _call_granite(prompt, max_tokens=600)


def get_energy_optimization_insights(
    system_summary: list,
    demand_stats: dict,
    load_factor: float,
    peak_hours: list,
) -> str:
    """
    Ask IBM Granite to generate energy optimization insights.
    """
    peak_hours_str = ", ".join(
        f"{p['hour']}:00 ({p['time_label']})" for p in peak_hours[:3]
    )

    sys_desc = "\n".join(
        f"  • {s['system']}: avg {s['avg_kw']} kW, total {s['total_kwh']} kWh"
        for s in system_summary
    )

    prompt = f"""You are an AI energy efficiency consultant specializing in commercial buildings.

=== BUILDING METRICS ===
Load Factor: {load_factor} (ideal > 0.75)
Peak Demand: {demand_stats['peak_kw']} kW
Average Demand: {demand_stats['average_kw']} kW
Baseline Demand: {demand_stats['baseline_kw']} kW
Peak Hours: {peak_hours_str}

=== SYSTEM ENERGY USE ===
{sys_desc}

=== TASK ===
Generate 5 specific energy optimization strategies to:
1. Reduce peak demand charges (demand charge reduction)
2. Improve the load factor from {load_factor} toward 0.85+
3. Identify waste reduction opportunities per system
4. Suggest renewable energy or storage integration opportunities
5. Provide estimated % savings potential for each strategy

Energy Optimization Insights:"""

    return _call_granite(prompt, max_tokens=650)


def get_anomaly_explanation(peak_event: dict) -> str:
    """
    Ask IBM Granite to explain a specific anomalous load spike.
    """
    systems = ", ".join(peak_event.get("systems_contributing", ["Unknown"]))
    prompt = f"""As a building energy expert, explain this abnormal electricity spike:

Timestamp: {peak_event['timestamp']}
Total Load: {peak_event['total_kw']} kW
Severity: {peak_event['severity']}
Anomaly Z-Score: {peak_event['z_score']}
Systems Contributing: {systems}

Provide:
1. Likely cause of this spike
2. Immediate action to take
3. Prevention strategy

Analysis:"""

    return _call_granite(prompt, max_tokens=350)
