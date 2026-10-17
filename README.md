# 📡 QueuePulse — Real-Time AI for Invisible Service Delays

**Predictive Queue & Congestion Decision-Support System**

## 🚨 Problem Statement
Public and essential-service facilities (hospitals, campus offices, banking halls, and government service centers) often experience sudden congestion, but managers only discover the problem after queues become unmanageable. Traditional queue monitors only report *current* headcount without predicting whether operations are about to cross a critical wait-time threshold.

**QueuePulse** is a real-time AI decision-support prototype that continuously analyzes arrival rates, service rates, queue growth, and counter utilization to:
1. **Predict** emerging congestion 5 minutes ahead before queues become critical.
2. **Diagnose** the primary root cause (service slowdown vs. arrival surge vs. backlog).
3. **Simulate What-If Interventions** and recommend targeted operational actions.
4. **Keep Humans in the Loop** with manager approval/rejection logging and CSV audit trails.

---

## 🏗️ System Architecture

```text
Streaming Queue Events (Arrivals, Service Rates, Queue State)
      │
      ▼
Real-Time Feature Engineering (Utilization, Net Queue Growth)
      │
      ├──► Random Forest Classifier ──► Congestion Risk Probability (LOW / MEDIUM / HIGH)
      │
      ├──► Random Forest Regressor  ──► 5-Minute Future Waiting Time Forecast (Minutes)
      │
      ▼
Root-Cause & What-If Decision Engine
      │
      ▼
Human-in-the-Loop Manager Approval (Add Staff / Redirect Arrivals / Monitor)
      │
      ▼
Closed-Loop Simulation & Audit Log Export (CSV)
