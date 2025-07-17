import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# Load model and encoders
model = joblib.load("model/career_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
mlb = joblib.load("model/mlb_encoder.pkl")

# Skill map for careers
career_skill_map = {
    "AI Engineer": ["Python", "TensorFlow"],
    "Data Scientist": ["Python", "SQL"],
    "Web Developer": ["HTML", "CSS", "JavaScript"],
    "App Developer": ["Java", "Kotlin"],
    "UX Designer": ["Creativity", "Figma"],
    "Cloud Engineer": ["AWS", "DevOps"],
    "Cybersecurity Analyst": ["Networking", "Linux"],
    "Educator": ["Communication", "Patience"],
    "Graphic Designer": ["Photoshop", "Illustrator"],
    "Game Developer": ["Unity", "C#"],
}

# Descriptions and learning resources
career_details = {
    "AI Engineer": {
        "desc": "Designs intelligent systems using machine learning and deep learning.",
        "learn": ["Python", "TensorFlow", "Neural Networks"],
        "link": "https://www.coursera.org/learn/machine-learning"
    },
    "Data Scientist": {
        "desc": "Analyzes data to extract insights using statistics and programming.",
        "learn": ["Python", "SQL", "Pandas"],
        "link": "https://www.kaggle.com/learn/data-science"
    },
    "Web Developer": {
        "desc": "Builds and maintains websites and web apps.",
        "learn": ["HTML", "CSS", "JavaScript"],
        "link": "https://www.freecodecamp.org/learn"
    },
    "App Developer": {
        "desc": "Creates mobile applications for Android or iOS.",
        "learn": ["Java", "Kotlin", "Android Studio"],
        "link": "https://developer.android.com/courses"
    },
    "UX Designer": {
        "desc": "Designs user-friendly interfaces and experiences.",
        "learn": ["Figma", "Creativity", "Design Thinking"],
        "link": "https://www.coursera.org/specializations/ux-design"
    },
    "Cloud Engineer": {
        "desc": "Manages cloud infrastructure and services.",
        "learn": ["AWS", "DevOps", "Docker"],
        "link": "https://www.udemy.com/course/aws-certified-cloud-practitioner/"
    },
    "Cybersecurity Analyst": {
        "desc": "Protects systems and networks from digital threats.",
        "learn": ["Networking", "Linux", "Ethical Hacking"],
        "link": "https://www.coursera.org/specializations/ibm-cybersecurity-analyst"
    },
    "Educator": {
        "desc": "Teaches students in academic or training settings.",
        "learn": ["Communication", "Patience", "Teaching Methodology"],
        "link": "https://www.coursera.org/learn/learning-how-to-learn"
    },
    "Graphic Designer": {
        "desc": "Creates visual designs using digital tools.",
        "learn": ["Photoshop", "Illustrator", "Creativity"],
        "link": "https://www.udemy.com/course/graphic-design-for-beginners/"
    },
    "Game Developer": {
        "desc": "Develops interactive games using game engines.",
        "learn": ["Unity", "C#", "Game Physics"],
        "link": "https://learn.unity.com/"
    },
}

# -------------------- UI -----------------------
st.set_page_config(page_title="CareerSense", page_icon="🎯")
st.title("🎯 CareerSense – Smart Career Recommendation System")
st.markdown("Find your ideal career path based on your interests, skills, and subject performance!")

st.header("🧠 Enter Your Details")

# --- Inputs ---
interests = st.text_input("🎯 Your Interest (e.g., AI Engineer, Web Developer)")
skills = st.text_input("🛠️ Your Skills (separate with `;`) e.g., Python;SQL;HTML")
maths = st.slider("📊 Maths Marks", 0, 100, 70)
cs = st.slider("💻 Computer Science Marks", 0, 100, 70)
english = st.slider("🗣️ English Marks", 0, 100, 70)

# ------------------ Predict -------------------
if st.button("🚀 Predict Career Path"):
    try:
        user_skills = [s.strip() for s in skills.split(";") if s.strip()]
        user_interests = [interests.strip()]

        # One-hot encode
        user_encoded = pd.DataFrame([0]*len(feature_columns), index=feature_columns).T
        for col in user_skills + user_interests:
            if col in user_encoded.columns:
                user_encoded.at[0, col] = 1

        user_encoded["Maths"] = maths
        user_encoded["CS"] = cs
        user_encoded["English"] = english

        # Prediction
        prediction = model.predict(user_encoded)[0]
        st.success(f"🎯 Suggested Career Path: **{prediction}**")

        # Career description + resource
        if prediction in career_details:
            st.markdown(f"**🔍 About {prediction}:** {career_details[prediction]['desc']}")
            st.markdown(f"**📘 Learn:** {', '.join(career_details[prediction]['learn'])}")
            st.markdown(f"[📚 Learn More]({career_details[prediction]['link']})")

        # ---------------- Feedback Engine ----------------
        st.markdown("### 📌 Personalized Feedback")

        if maths < 60:
            st.warning("📉 Your Maths marks are low. Work on problem-solving and logical reasoning.")
        if cs < 60:
            st.warning("📉 Your CS marks are low. Strengthen your programming and algorithms.")
        if english < 60:
            st.warning("📉 English score is low. Improve your writing and communication skills.")

        # Interest-to-skill gap
        if interests.strip() in career_skill_map:
            required = career_skill_map[interests.strip()]
            missing = [s for s in required if s not in user_skills]
            if missing:
                st.warning(f"🎯 To pursue **{interests.strip()}**, consider learning: **{', '.join(missing)}**")
            else:
                st.success(f"✅ You already have strong skills aligned with your interest in **{interests.strip()}**!")

        # Alternate role suggestions
        st.markdown("### 💡 You might also be great at:")
        found = False
        match_scores = []

        for career, req_skills in career_skill_map.items():
            if career != prediction:
                matched = [s for s in user_skills if s in req_skills]
                score = round(len(matched) / len(req_skills) * 100)
                if len(matched) >= 2:
                    found = True
                    match_scores.append({"Career": career, "Match %": score})
                    st.info(f"🔹 **{career}** – {score}% match based on: {', '.join(matched)}")

        if not found:
            st.write("No strong alternate matches found — but you're on the right path!")

        # Career Match Bar Chart (Top 3)
        if match_scores:
            top_matches = sorted(match_scores, key=lambda x: x["Match %"], reverse=True)[:3]
            fig = px.bar(top_matches, x="Career", y="Match %", color="Career", title="Top Alternate Career Matches")
            st.plotly_chart(fig)

        # Learning Plan
        st.markdown("### 🧠 Personalized Learning Plan")
        if prediction in career_details:
            for i, topic in enumerate(career_details[prediction]['learn'], 1):
                st.markdown(f"{i}. {topic}")

    except Exception as e:
        st.error("❌ Something went wrong.")
        st.exception(e)
