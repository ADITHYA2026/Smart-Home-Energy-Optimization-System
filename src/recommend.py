import pandas as pd
import numpy as np


# -------------------------------------------------------
# PRIORITY CLASSIFICATION (SMARTER)
# -------------------------------------------------------
def classify_priority(value, avg_usage):
    high_threshold = avg_usage * 1.45
    medium_threshold = avg_usage * 1.15

    if value >= high_threshold:
        return "HIGH"
    elif value >= medium_threshold:
        return "MEDIUM"
    return "LOW"


# -------------------------------------------------------
# ADVANCED RICH RECOMMENDATION ENGINE
# -------------------------------------------------------
def generate_recommendation_row(row, avg_usage):
    usage = row["Hybrid"]
    sub1, sub2, sub3 = row["Sub_metering_1"], row["Sub_metering_2"], row["Sub_metering_3"]
    voltage = row["Voltage"]

    hour = row["hour"]
    weekday = row["weekday"]
    season = row["season"]

    rec = []
    blocks = []

    # -------------------------------------------------------
    # 1. HIGH-LEVEL ENERGY SUMMARY
    # -------------------------------------------------------
    summary = f"""
### 🔍 Energy Usage Summary
- Predicted consumption: **{usage:.2f} kW**
- Daily average consumption: **{avg_usage:.2f} kW**
- Voltage level: **{voltage} V**
"""
    blocks.append(summary)

    # -------------------------------------------------------
    # 2. USAGE CATEGORY
    # -------------------------------------------------------
    if usage >= avg_usage * 1.5:
        level = "🔴 Critical Load"
        rec.append("Immediate reduction recommended — avoid all heavy appliances (AC, heater, dryer).")
    elif usage >= avg_usage * 1.1:
        level = "🟠 Elevated Load"
        rec.append("Moderate load — delay non-essential appliances to save cost.")
    else:
        level = "🟢 Normal Load"
        rec.append("Low load — perfect time to use appliances efficiently.")

    blocks.append(f"### ⚡ Load Classification\n- **{level}**")

    # -------------------------------------------------------
    # 3. APPLIANCE INTELLIGENCE
    # -------------------------------------------------------
    appliance_insights = []

    if sub3 > 25:
        appliance_insights.append("🔥 *High HVAC/Heater usage detected — Optimize thermostat by +2°C / -2°C.*")

    if sub2 > 18:
        appliance_insights.append("🌀 Laundry spikes — Run washing machine after 10 PM or before 6 AM.")

    if sub1 > 12:
        appliance_insights.append("🍳 Kitchen usage high — Avoid using microwave + induction together.")

    if not appliance_insights:
        appliance_insights.append("✔️ All appliance loads are within normal range.")

    blocks.append("### 🛠 Appliance Insights\n" + "\n".join(f"- {x}" for x in appliance_insights))

    # -------------------------------------------------------
    # 4. SAFETY CHECKS
    # -------------------------------------------------------
    safety = []
    if voltage < 225:
        safety.append("⚠️ Low voltage — Avoid sensitive electronics; risk of under-voltage damage.")
    if voltage > 245:
        safety.append("⚠️ High voltage — Ensure stabilizer is functioning properly.")

    if not safety:
        safety.append("✔️ Voltage stable — All appliances safe to operate.")

    blocks.append("### 🛡 Safety Diagnostics\n" + "\n".join(f"- {x}" for x in safety))

    # -------------------------------------------------------
    # 5. TIME-OF-DAY ANALYTICS
    # -------------------------------------------------------
    time_insights = []
    if 18 <= hour <= 22:
        time_insights.append("🌆 Evening peak — Avoid AC, geysers, and ovens now.")
    elif 2 <= hour <= 6:
        time_insights.append("🌙 Off-peak tariff — Ideal for laundry/dishwasher cycles.")
    else:
        time_insights.append("⏱ Balanced load hours — No restrictions.")

    blocks.append("### 🕒 Time-of-Day Insights\n" + "\n".join(f"- {x}" for x in time_insights))

    # -------------------------------------------------------
    # 6. WEEKDAY / WEEKEND PATTERN
    # -------------------------------------------------------
    if weekday >= 5:
        blocks.append("### 📅 Weekend Pattern\n- Energy usage tends to rise — Batch cooking/laundry recommended.")
    else:
        blocks.append("### 📅 Weekday Pattern\n- Normal consumption day — No special adjustments needed.")

    # -------------------------------------------------------
    # 7. SEASONAL EFFECTS
    # -------------------------------------------------------
    if season == 1:
        blocks.append("### ❄ Winter Advisory\n- Expect heater load — Monitor Sub_metering_3 closely.")
    elif season == 3:
        blocks.append("### ☀ Summer Advisory\n- AC consumption will rise — Maintain 24–26°C thermostat settings.")
    else:
        blocks.append("### 🍃 Seasonal Note\n- Energy demand moderate — Ideal conditions.")

    # -------------------------------------------------------
    # 8. SPIKE DETECTION
    # -------------------------------------------------------
    if abs(row["Hybrid"] - row["Actual"]) > 1.5:
        blocks.append("### 🚨 Spike Detected\n- Sudden consumption anomaly — Check for appliances left ON.")

    # -------------------------------------------------------
    # 9. COST ESTIMATION
    # -------------------------------------------------------
    estimated_cost = usage * 8.2  # ₹ per kWh approx
    blocks.append(
        f"### 💰 Estimated Cost\n- **₹{estimated_cost:.2f} per hour** at current usage level."
    )

    # -------------------------------------------------------
    # 10. GENERAL ENERGY SAVING TIPS
    # -------------------------------------------------------
    blocks.append("""
### 💡 General Energy Tips
- Use LED bulbs & 5-star appliances  
- Clean AC filters monthly  
- Use sunlight & natural ventilation  
- Turn off standby devices to save 5–10% energy  
""")

    # -------------------------------------------------------
    return "\n\n".join(blocks)


# -------------------------------------------------------
# PIPELINE
# -------------------------------------------------------
def generate_recommendations():
    print("Loading hybrid predictions...")
    df = pd.read_csv("data/processed/hybrid_results.csv")

    print("Loading feature dataset...")
    feat = pd.read_csv("data/processed/features.csv", index_col="DateTime", parse_dates=True)
    feat = feat.iloc[60:].reset_index()

    df = df.reset_index(drop=True)
    df = pd.concat([feat, df], axis=1)

    avg_usage = df["Hybrid"].mean()

    print("Assigning priorities...")
    df["Priority"] = df["Hybrid"].apply(lambda v: classify_priority(v, avg_usage))

    print("Generating EXTENDED GRAND recommendations...")
    df["Recommendation"] = df.apply(lambda row: generate_recommendation_row(row, avg_usage), axis=1)

    df.to_csv("data/processed/final_recommendations.csv", index=False)

    print("\n🎉 ENTERPRISE Recommendations saved → data/processed/final_recommendations.csv")
    return df


if __name__ == "__main__":
    generate_recommendations()