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
# 3. STREAMING SIMULATION & STATE FUNCTIONS
# --------------------------------------------------
def get_all_counter_rates():
    """Calculate effective arrival and service rates across all 3 counters."""
    raw_arrivals = []
    raw_services = []

    for i in range(3):
        arrival = st.session_state.base_arrivals[i]
        service = st.session_state.base_service[i]

        # Controlled failure scenario at Counter 2 (index 1).
        if i == 1 and st.session_state.scenario_slowdown:
            if st.session_state.minute >= 8:
                service = 0.8

        # Controlled arrival surge across counters.
        if st.session_state.scenario_surge:
            if st.session_state.minute >= 12:
                arrival += [0.5, 1.1, 0.3][i]

        # Apply human-approved staffing interventions.
        active_actions = st.session_state.approved.get(i + 1, [])
        if "Add staff" in active_actions:
            service += 2.5

        raw_arrivals.append(arrival)
        raw_services.append(service)

    # Apply arrival redirection from congested counters to the shortest queue.
    effective_arrivals = list(raw_arrivals)
    redirected_total = 0.0
    redirected_sources = []

    for i in range(3):
        active_actions = st.session_state.approved.get(i + 1, [])
        if "Redirect arrivals" in active_actions:
            cut = effective_arrivals[i] * 0.45
            effective_arrivals[i] -= cut
            redirected_total += cut
            redirected_sources.append(i)

    if redirected_total > 0:
        # Route redirected arrivals to the counter with the shortest queue
        # that is not itself redirecting arrivals.
        candidate_indices = [
            idx for idx in range(3) if idx not in redirected_sources
        ] or [0, 1, 2]
        target_i = min(
            candidate_indices,
            key=lambda idx: st.session_state.queues[idx]
        )
        effective_arrivals[target_i] += redirected_total

    return list(zip(effective_arrivals, raw_services))


def get_current_state():
    state = []
    all_rates = get_all_counter_rates()

    for i in range(3):
        arrival_rate, service_rate = all_rates[i]
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

        # Root-cause priority: check service slowdown FIRST so it isn't masked
        # by arrival_rate > service_rate.
        if service_rate <= 1.0 and arrival_rate > 3.0:
            cause = "Service rate dropped critically at counter (compounded by arrival surge)"
        elif service_rate <= 1.0:
            cause = "Service rate has dropped critically at counter"
        elif arrival_rate > service_rate * 1.15:
            cause = "Arrival demand surge exceeds service capacity"
        elif queue >= 15:
            cause = "Existing queue backlog is overloaded"
        else:
            cause = "Demand is approaching available capacity"

        # Approximate time until the queue reaches the 12-minute threshold.
        target_queue = CRITICAL_WAIT * service_rate

        if predicted_wait >= CRITICAL_WAIT:
            time_to_critical = 0.0
        elif growth > 0:
            time_to_critical = max(
                0.0, (target_queue - queue) / growth
            )
        else:
            time_to_critical = None

        # Estimate an intervention using a what-if calculation (+2.5/min staff boost).
        improved_service = service_rate + 2.5
        current_future_queue = max(0, queue + 5 * growth)
        improved_future_queue = max(
            0, queue + 5 * (arrival_rate - improved_service)
        )

        current_wait = current_future_queue / max(service_rate, 0.1)
        improved_wait = improved_future_queue / max(
            improved_service, 0.1
        )

        reduction = max(0.0, current_wait - improved_wait)

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
            "estimated_reduction": reduction,
            "active_interventions": st.session_state.approved.get(i + 1, [])
        })

    return state


def record_snapshot():
    snapshot = get_current_state()
    for row in snapshot:
        st.session_state.history.append({
            "Minute": st.session_state.minute,
            "Counter": f"Counter {row['counter']}",
            "Queue": row["queue_length"],
            "Predicted wait": row["predicted_wait"],
            "Risk": row["risk"]
        })


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
    record_snapshot()


if "minute" not in st.session_state:
    initialize()


def advance_one_minute():
    st.session_state.minute += 1
    all_rates = get_all_counter_rates()

    for i in range(3):
        arrival_rate, service_rate = all_rates[i]
        incoming = np.random.poisson(arrival_rate)
        completed = np.random.poisson(service_rate)

        st.session_state.queues[i] = max(
            0,
            int(st.session_state.queues[i] + incoming - completed)
        )

    record_snapshot()


# --------------------------------------------------
# 4. SIDEBAR CONTROLS
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
    initialize()
    st.rerun()

# --------------------------------------------------
# 5. LIVE DASHBOARD
# --------------------------------------------------
state = get_current_state()

high_count = sum(row["risk"] == "HIGH" for row in state)
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

        if row["active_interventions"]:
            st.info(
                "Active: " + ", ".join(row["active_interventions"])
            )

# --------------------------------------------------
# 6. FORECAST AND ROOT CAUSE
# --------------------------------------------------
st.divider()
st.header("🔎 Forecast & Root-Cause Analysis")

any_flagged = False
for row in state:
    if row["risk"] != "LOW":
        any_flagged = True
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
                "the queue is stabilizing or shrinking under current capacity."
            )

        st.write(
            f"**What-if estimate:** Adding service capacity (+2.5/min) could "
            f"reduce the five-minute waiting-time estimate by about "
            f"{row['estimated_reduction']:.1f} minutes."
        )

if not any_flagged:
    st.success("All counters are currently operating within normal wait-time thresholds.")

# --------------------------------------------------
# 7. STREAMING CHART
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

# --------------------------------------------------
# 8. HUMAN APPROVAL OF RECOMMENDATIONS
# --------------------------------------------------
st.divider()
st.header("🧑‍💼 Human-in-the-Loop Decision Center")
st.write(
    "QueuePulse recommends actions. A manager must approve an "
    "intervention before it is applied to the simulation."
)

# Automatically default to the counter with the longest predicted wait.
worst_counter_index = int(
    np.argmax([row["predicted_wait"] for row in state])
)

counter_choice = st.selectbox(
    "Select counter",
    [1, 2, 3],
    index=worst_counter_index,
    format_func=lambda x: f"Counter {x}"
)

action_choice = st.selectbox(
    "Recommended intervention",
    [
        "Add staff",
        "Redirect arrivals",
        "Add staff + Redirect arrivals",
        "Monitor only"
    ]
)

col_a, col_b = st.columns(2)

with col_a:
    if st.button("✅ Approve action", use_container_width=True):
        if action_choice != "Monitor only":
            current_list = list(
                st.session_state.approved.get(counter_choice, [])
            )
            if action_choice == "Add staff + Redirect arrivals":
                for act in ["Add staff", "Redirect arrivals"]:
                    if act not in current_list:
                        current_list.append(act)
            else:
                if action_choice not in current_list:
                    current_list.append(action_choice)
            st.session_state.approved[counter_choice] = current_list

        st.session_state.feedback.append({
            "Minute": st.session_state.minute,
            "Counter": counter_choice,
            "Decision": action_choice,
            "Status": "Approved"
        })
        st.rerun()

with col_b:
    if st.button("❌ Reject action", use_container_width=True):
        st.session_state.feedback.append({
            "Minute": st.session_state.minute,
            "Counter": counter_choice,
            "Decision": action_choice,
            "Status": "Rejected"
        })
        st.rerun()

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
# 9. MODEL EVALUATION
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