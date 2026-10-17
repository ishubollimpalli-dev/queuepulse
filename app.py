
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="QueuePulse | Predictive Operations",
    page_icon="📡",
    layout="wide"
)

st.title("📡 QueuePulse")
st.subheader("Real-Time AI for Invisible Service Delays")
st.caption(
    "Predict emerging bottlenecks • Explain causes • "
    "Recommend actions • Keep managers in control"
)

CRITICAL_WAIT = 12.0
FEATURES = [
    "queue_length",
    "arrival_rate",
    "service_rate",
    "queue_growth",
    "utilization"
]

# --------------------------------------------------
# 2. TRAIN MODELS ON SYNTHETIC DATA
# --------------------------------------------------
@st.cache_resource
def train_models():
    rng = np.random.default_rng(42)
    n = 12000

    queue = rng.integers(0, 81, n).astype(float)
    arrivals = rng.uniform(0.2, 6.0, n)
    service = rng.uniform(0.3, 6.5, n)

    growth = arrivals - service
    utilization = arrivals / np.maximum(service, 0.1)

    # Approximate queue and waiting time five minutes ahead.
    future_queue = np.maximum(0, queue + 5 * growth)
    future_wait = future_queue / np.maximum(service, 0.1)
    congestion = (future_wait >= CRITICAL_WAIT).astype(int)

    X = pd.DataFrame({
        "queue_length": queue,
        "arrival_rate": arrivals,
        "service_rate": service,
        "queue_growth": growth,
        "utilization": utilization
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X, congestion,
        test_size=0.2,
        random_state=42,
        stratify=congestion
    )

    classifier = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    classifier.fit(X_train, y_train)

    wait_train_X, wait_test_X, wait_train_y, wait_test_y = (
        train_test_split(
            X, future_wait,
            test_size=0.2,
            random_state=42
        )
    )

    regressor = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    regressor.fit(wait_train_X, wait_train_y)

    accuracy = accuracy_score(
        y_test, classifier.predict(X_test)
    )
    wait_mae = mean_absolute_error(
        wait_test_y, regressor.predict(wait_test_X)
    )

    return classifier, regressor, accuracy, wait_mae


classifier, regressor, model_accuracy, model_mae = train_models()

# --------------------------------------------------
# 3. SESSION STATE / INITIAL QUEUES
# --------------------------------------------------
def initialize():
    st.session_state.minute = 0
    st.session_state.queues = [8, 5, 3]
    st.session_state.base_arrivals = [2.0, 2.4, 1.5]
    st.session_state.base_service = [3.0, 3.0, 2.7]
    st.session_state.history = []
    st.session_state.feedback = []
    st.session_state.approved = {}
    st.session_state.scenario_slowdown = True
    st.session_state.scenario_surge = True


if "minute" not in st.session_state:
    initialize()

# --------------------------------------------------
# 4. STREAMING SIMULATION
# --------------------------------------------------
def get_counter_rates(i):
    arrival = st.session_state.base_arrivals[i]
    service = st.session_state.base_service[i]

    # Controlled failure scenario for the demo.
    if i == 1 and st.session_state.scenario_slowdown:
        if st.session_state.minute >= 8:
            service = 0.8

    if st.session_state.scenario_surge:
        if st.session_state.minute >= 12:
            arrival += [0.5, 1.1, 0.3][i]

    # Human-approved interventions.
    intervention = st.session_state.approved.get(i + 1)

    if intervention == "Add staff":
        service += 1.5

    return arrival, service


def advance_one_minute():
    st.session_state.minute += 1

    rates = []
    for i in range(3):
        arrival_rate, service_rate = get_counter_rates(i)

        # Redirect some new arrivals from a congested counter
        # to the least-loaded available counter.
        if st.session_state.approved.get(i + 1) == "Redirect arrivals":
            arrival_rate *= 0.55

        rates.append((arrival_rate, service_rate))

    # Identify the counter with the shortest queue.
    target_i = min(range(3), key=lambda i: st.session_state.queues[i])

    for i in range(3):
        arrival_rate, service_rate = rates[i]

        # Redirected people are routed to the shortest queue.
        if any(
            action == "Redirect arrivals"
            for action in st.session_state.approved.values()
        ) and i == target_i:
            arrival_rate += 0.8

        incoming = np.random.poisson(arrival_rate)
        completed = np.random.poisson(service_rate)

        st.session_state.queues[i] = max(
            0,
            int(st.session_state.queues[i] + incoming - completed)
        )

    snapshot = get_current_state()

    for row in snapshot:
        st.session_state.history.append({
            "Minute": st.session_state.minute,
            "Counter": f"Counter {row['counter']}",
            "Queue": row["queue_length"],
            "Predicted wait": row["predicted_wait"],
            "Risk": row["risk"]
        })


def get_current_state():
    state = []

    for i in range(3):
        arrival_rate, service_rate = get_counter_rates(i)
        queue = st.session_state.queues[i]

        growth = arrival_rate - service_rate
        utilization = arrival_rate / max(service_rate, 0.1)

        features = pd.DataFrame([{
            "queue_length": queue,
            "arrival_rate": arrival_rate,
            "service_rate": service_rate,
            "queue_growth": growth,
            "utilization": utilization
        }])[FEATURES]

        probability = classifier.predict_proba(features)[0]
        classes = list(classifier.classes_)
        risk_probability = (
            float(probability[classes.index(1)])
            if 1 in classes else 0.0
        )

        predicted_wait = max(
            0.0, float(regressor.predict(features)[0])
        )

        # Explicit waiting-time guardrail.
        if predicted_wait >= CRITICAL_WAIT:
            risk = "HIGH"
        elif predicted_wait >= 8 or risk_probability >= 0.40:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        if arrival_rate > service_rate * 1.15:
            cause = "Arrival demand exceeds service capacity"
        elif service_rate < 1.0:
            cause = "Service rate has dropped"
        elif queue >= 15:
            cause = "Existing queue is overloaded"
        else:
            cause = "Demand is approaching available capacity"

        # Approximate time until the queue reaches the
        # 12-minute waiting-time threshold.
        target_queue = CRITICAL_WAIT * service_rate

        if predicted_wait >= CRITICAL_WAIT:
            time_to_critical = 0.0
        elif growth > 0:
            time_to_critical = max(
                0.0, (target_queue - queue) / growth
            )
        else:
            time_to_critical = None

        # Estimate an intervention using a simple what-if model.
        improved_service = service_rate + 1.5
        current_future_queue = max(0, queue + 5 * growth)
        improved_future_queue = max(
            0, queue + 5 * (arrival_rate - improved_service)
        )

        current_wait = current_future_queue / max(service_rate, 0.1)
        improved_wait = improved_future_queue / max(
            improved_service, 0.1
        )

        reduction = max(0, current_wait - improved_wait)

        state.append({
            "counter": i + 1,
            "queue_length": queue,
            "arrival_rate": arrival_rate,
            "service_rate": service_rate,
            "queue_growth": growth,
            "utilization": utilization,
            "predicted_wait": predicted_wait,
            "risk_probability": risk_probability,
            "risk": risk,
            "cause": cause,
            "time_to_critical": time_to_critical,
            "estimated_reduction": reduction
        })

    return state


# --------------------------------------------------
# 5. SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("Simulation Controls")
st.sidebar.write("One simulation step represents one virtual minute.")

st.session_state.scenario_slowdown = st.sidebar.checkbox(
    "Inject service slowdown at Counter 2",
    value=st.session_state.scenario_slowdown
)
st.session_state.scenario_surge = st.sidebar.checkbox(
    "Simulate an arrival surge",
    value=st.session_state.scenario_surge
)

if st.sidebar.button("▶ Advance one minute", use_container_width=True):
    advance_one_minute()

if st.sidebar.button("⏩ Simulate 10 minutes", use_container_width=True):
    for _ in range(10):
        advance_one_minute()

if st.sidebar.button("↻ Reset simulation", use_container_width=True):
    for key in [
        "minute", "queues", "base_arrivals", "base_service",
        "history", "feedback", "approved",
        "scenario_slowdown", "scenario_surge"
    ]:
        st.session_state.pop(key, None)
    initialize()
    st.rerun()

# --------------------------------------------------
# 6. LIVE DASHBOARD
# --------------------------------------------------
state = get_current_state()

high_count = sum(row["risk"] == "HIGH" for row in state)
max_queue = max(row["queue_length"] for row in state)
average_wait = np.mean([row["predicted_wait"] for row in state])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Virtual minute", st.session_state.minute)
m2.metric("Total people waiting", sum(st.session_state.queues))
m3.metric("High-risk counters", high_count)
m4.metric("Average predicted wait", f"{average_wait:.1f} min")

st.divider()
st.header("Live Counter Status")

cards = st.columns(3)

for i, row in enumerate(state):
    with cards[i]:
        st.subheader(f"Counter {row['counter']}")

        if row["risk"] == "HIGH":
            st.error("🔴 HIGH RISK")
        elif row["risk"] == "MEDIUM":
            st.warning("🟠 MEDIUM RISK")
        else:
            st.success("🟢 LOW RISK")

        st.metric("Queue length", row["queue_length"])
        st.metric("Predicted wait", f"{row['predicted_wait']:.1f} min")
        st.metric("Arrival rate", f"{row['arrival_rate']:.1f}/min")
        st.metric("Service rate", f"{row['service_rate']:.1f}/min")
        st.progress(min(1.0, row["utilization"] / 2.0))
        st.caption(f"Utilization: {row['utilization'] * 100:.0f}%")

# --------------------------------------------------
# 7. FORECAST AND ROOT CAUSE
# --------------------------------------------------
st.divider()
st.header("🔎 Forecast & Root-Cause Analysis")

for row in state:
    if row["risk"] != "LOW":
        st.markdown(f"### Counter {row['counter']}")

        st.write(f"**Primary cause:** {row['cause']}")
        st.write(
            f"**Model-estimated congestion probability:** "
            f"{row['risk_probability'] * 100:.1f}%"
        )

        if row["time_to_critical"] is not None:
            st.write(
                f"**Estimated time until critical threshold:** "
                f"{row['time_to_critical']:.1f} virtual minutes "
                "(approximate queue-growth calculation)"
            )
        else:
            st.write(
                "**Time to critical:** Not currently estimated; "
                "the queue is not growing under the current rate estimate."
            )

        st.write(
            f"**What-if estimate:** Adding service capacity could "
            f"reduce the five-minute waiting-time estimate by about "
            f"{row['estimated_reduction']:.1f} minutes."
        )

# --------------------------------------------------
# 8. STREAMING CHART
# --------------------------------------------------
st.divider()
st.header("📈 Queue History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)

    fig = px.line(
        history_df,
        x="Minute",
        y="Queue",
        color="Counter",
        markers=True,
        title="Queue Length Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)

    wait_fig = px.line(
        history_df,
        x="Minute",
        y="Predicted wait",
        color="Counter",
        markers=True,
        title="Predicted Waiting Time Over Time"
    )
    st.plotly_chart(wait_fig, use_container_width=True)
else:
    st.info("Advance the simulation to start collecting queue history.")

# --------------------------------------------------
# 9. HUMAN APPROVAL OF RECOMMENDATIONS
# --------------------------------------------------
st.divider()
st.header("🧑‍💼 Human-in-the-Loop Decision Center")
st.write(
    "QueuePulse recommends actions. A manager must approve an "
    "intervention before it is applied to the simulation."
)

counter_choice = st.selectbox(
    "Select counter",
    [1, 2, 3],
    format_func=lambda x: f"Counter {x}"
)

action_choice = st.selectbox(
    "Recommended intervention",
    ["Add staff", "Redirect arrivals", "Monitor only"]
)

col_a, col_b = st.columns(2)

with col_a:
    if st.button("✅ Approve action"):
        if action_choice == "Monitor only":
            st.session_state.feedback.append({
                "Minute": st.session_state.minute,
                "Counter": counter_choice,
                "Decision": "Monitor only",
                "Status": "Approved"
            })
        else:
            st.session_state.approved[counter_choice] = action_choice
            st.session_state.feedback.append({
                "Minute": st.session_state.minute,
                "Counter": counter_choice,
                "Decision": action_choice,
                "Status": "Approved"
            })
        st.success("Decision recorded. The simulation will use this action.")

with col_b:
    if st.button("❌ Reject action"):
        st.session_state.feedback.append({
            "Minute": st.session_state.minute,
            "Counter": counter_choice,
            "Decision": action_choice,
            "Status": "Rejected"
        })
        st.warning("Rejection recorded. No new intervention was applied.")

if st.session_state.feedback:
    st.subheader("Decision and Feedback Log")
    display_df = pd.DataFrame(st.session_state.feedback)
    st.dataframe(display_df, use_container_width=True)

    st.download_button(
        "Download decision log (CSV)",
        data=display_df.to_csv(index=False),
        file_name="queuepulse_decision_log.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# 10. MODEL EVALUATION
# --------------------------------------------------
with st.expander("Model evaluation and technical details"):
    st.write(
        "Both models are trained on synthetic data generated by "
        "QueuePulse. These scores are not evidence of real-world "
        "performance."
    )
    st.metric(
        "Held-out synthetic classification accuracy",
        f"{model_accuracy * 100:.1f}%"
    )
    st.metric(
        "Held-out synthetic waiting-time MAE",
        f"{model_mae:.2f} min"
    )

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": classifier.feature_importances_
    }).sort_values("Importance", ascending=False)

    st.write("Classifier feature importance")
    st.dataframe(importance_df, use_container_width=True)

st.caption(
    "QueuePulse is a simulation-based decision-support prototype. "
    "All generated arrivals, predictions, and intervention effects "
    "must be validated against real observations before operational use."
)