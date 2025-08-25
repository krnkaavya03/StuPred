import streamlit as st 
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px

# =========================
# Load trained model
# =========================
model = pickle.load(open("student_model.pkl", "rb"))

# =========================
# Page Configuration
# =========================
st.set_page_config(page_title="StuPred", layout="wide", page_icon="🎓")
st.title("🎓 StuPred: Student Success Prediction Dashboard")

# =========================
# Custom Styling (Dark Mode Only)
# =========================
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #111111 !important;
    color: white !important;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #1c1c1c !important;
    color: white !important;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* KPI Metrics */
[data-testid="stMetric"] {
    background-color: #222222 !important;
    color: white !important;
    border-radius: 10px;
    padding: 10px;
}

/* Above deploy area */
footer, .viewerBadge_container__1QSob, #MainMenu {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("📝 Student Profile Inputs")

with st.sidebar.expander("📚 Academic Performance"):
    attendance = st.slider("Attendance (%)", 50, 100, 75)
    previous_grade = st.slider("Previous Grade (%)", 50, 100, 70)
    midterm_score = st.slider("Midterm Score (%)", 50, 100, 70)

with st.sidebar.expander("📖 Study Habits"):
    study_hours = st.number_input("Study Hours per Week", 10, 70, 30)
    assignments_done = st.slider("Assignments Completed", 5, 15, 10)

with st.sidebar.expander("🤝 Participation & Activities"):
    participations = st.slider("Class Participations", 1, 10, 5)
    active_extracurricular = st.slider("Extracurricular Activities", 1, 5, 2)

# =========================
# Predict Button
# =========================
if st.button("🚀 Predict Success", use_container_width=True):
    # Prepare features
    features = pd.DataFrame([[attendance, study_hours, assignments_done, previous_grade,
                              midterm_score, participations, active_extracurricular]],
                            columns=['attendance','study_hours','assignments_done','previous_grade',
                                     'midterm_score','participations','active_extracurricular'])
    
    prediction = model.predict(features)[0]
    prediction_proba = model.predict_proba(features)[0][1]

    # =========================
    # KPI Cards
    # =========================
    st.markdown("## 📊 Key Metrics")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Attendance", f"{attendance}%")
    kpi2.metric("Study Hours", f"{study_hours}/week")
    kpi3.metric("Assignments", assignments_done)
    kpi4.metric("Prev Grade", f"{previous_grade}%")

    kpi5, kpi6, kpi7 = st.columns(3)
    kpi5.metric("Midterm", f"{midterm_score}%")
    kpi6.metric("Participation", participations)
    kpi7.metric("Extracurricular", active_extracurricular)

    st.markdown("---")

    # =========================
    # Tabs for Charts
    # =========================
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Probability Gauge", "📡 Radar", "📊 Bar Chart", "🥧 Pie Chart", "💡 Scatter"]
    )

    # Gauge
    with tab1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba*100,
            title={'text': "Success Probability (%)"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#00cc96"},
                   'steps': [
                       {'range': [0, 50], 'color': "red"},
                       {'range': [50, 75], 'color': "yellow"},
                       {'range': [75, 100], 'color': "green"}]}))
        fig_gauge.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Radar
    with tab2:
        features_list = ['Attendance','Study Hours','Assignments','Previous Grade',
                         'Midterm','Participation','Extracurricular']
        values = [attendance, study_hours, assignments_done, previous_grade,
                  midterm_score, participations, active_extracurricular]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
              r=values,
              theta=features_list,
              fill='toself',
              name='Student Profile'
        ))
        fig_radar.update_layout(
          polar=dict(radialaxis=dict(visible=True, range=[0, max(values)+10])),
          template="plotly_dark"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Bar
    with tab3:
        bar_data = pd.DataFrame({
            'Feature': features_list,
            'Value': values,
            'Max': [100, 70, 15, 100, 100, 10, 5]
        })
        fig_bar = px.bar(bar_data, x='Feature', y='Value', text='Value',
                         range_y=[0, max(bar_data['Max'])],
                         color='Feature', template="plotly_dark",
                         title="📊 Student Profile Inputs")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Pie
    with tab4:
        fig_pie = px.pie(bar_data, values='Value', names='Feature', 
                         color_discrete_sequence=px.colors.sequential.Plasma,
                         title="🥧 Contribution of Each Factor")
        fig_pie.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Scatter
    with tab5:
        scatter_data = pd.DataFrame({
            'Study Hours': [study_hours],
            'Previous Grade': [previous_grade],
            'Success Prob': [prediction_proba*100]
        })
        fig_scatter = px.scatter(scatter_data, x='Study Hours', y='Previous Grade',
                                 size='Success Prob', color='Success Prob',
                                 color_continuous_scale='Plasma',
                                 size_max=40,
                                 template="plotly_dark",
                                 title="Study Hours vs Previous Grade (Bubble = Success Prob)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # =========================
    # Prediction Message
    # =========================
    st.markdown("## 📈 Prediction Result")
    if prediction == 1:
        st.success(f"✅ Student is likely to succeed! (Probability: {prediction_proba*100:.1f}%)")
        st.balloons()
    else:
        st.error(f"❌ Student may need improvement. (Probability: {prediction_proba*100:.1f}%)")

# =========================
# Show Input Profile
# =========================
st.markdown("---")
with st.expander("📝 View Input Profile"):
    st.write({
        "Attendance (%)": attendance,
        "Study Hours/Week": study_hours,
        "Assignments Done": assignments_done,
        "Previous Grade (%)": previous_grade,
        "Midterm Score (%)": midterm_score,
        "Participation": participations,
        "Extracurricular": active_extracurricular
    })
