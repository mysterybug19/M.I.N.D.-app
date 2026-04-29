import joblib
import pandas as pd
from xgboost import XGBClassifier as XGBC
from sklearn.multioutput import MultiOutputClassifier
import streamlit as st
import os

questions = [
    {"key": "age", "label": "Câți ani ai?", "type": "age"},
    {"key": "gender", "label": "Care este genul tău?", "type": "choice", "options": ["Male", "Female", "Other"]},
    {"key": "education_level", "label": "Care este nivelul tău de educație?", "type": "choice",
     "options": ["Fără studii", "Școală generală", "Liceu", "Studii superioare",
                 "Doctorat/Master avansat"]},
    {"key": "employment_status", "label": "Care este situația ta ocupațională actuală?", "type": "choice",
     "options": ["Employed", "Unemployed", "Student", "Retired", "Disabled"]},
    {"key": "family_history_mental_illness", "label": "Există în familia ta persoane diagnosticate cu o boală psihică?",
     "type": "yn"},
    {"key": "social_interaction_difficulty",
     "label": "Îți este greu să inițiezi sau să menții conversații cu alte persoane?", "type": "1-5"},
    {"key": "communication_difficulty", "label": "Ai dificultăți în a-ți exprima gândurile sau emoțiile verbal?",
     "type": "1-5"},
    {"key": "repetitive_behaviors",
     "label": "Ai comportamente sau mișcări pe care le repeți des (ex: legănare, aliniere de obiecte)?", "type": "1-5"},
    {"key": "sensory_sensitivity", "label": "Ești sensibil/ă la sunete puternice, lumini sau texturi?", "type": "1-5"},
    {"key": "eye_contact_difficulty",
     "label": "Îți este incomod sau dificil să menții contactul vizual în conversații?", "type": "1-5"},
    {"key": "routine_rigidity", "label": "Te simți deranjat/ă când rutina zilnică îți este schimbată?", "type": "1-5"},
    {"key": "special_interests_intensity",
     "label": "Ai interese foarte specifice și intense față de care petreci mult timp?", "type": "1-5"},
    {"key": "nonverbal_comm_difficulty",
     "label": "Îți este greu să interpretezi limbajul corpului sau expresiile faciale?", "type": "1-5"},
    {"key": "inattention", "label": "Îți este greu să rămâi concentrat/ă pe o sarcină pentru o perioadă mai lungă?",
     "type": "1-5"},
    {"key": "hyperactivity", "label": "Simți nevoia să fii mereu în mișcare sau să faci ceva cu mâinile?",
     "type": "1-5"},
    {"key": "impulsivity", "label": "Acționezi impulsiv, fără să te gândești la consecințe?", "type": "1-5"},
    {"key": "difficulty_completing_tasks", "label": "Ai dificultăți în a duce sarcinile până la capăt?", "type": "1-5"},
    {"key": "forgetfulness", "label": "Uiți frecvent lucruri importante (întâlniri, obiecte, sarcini)?", "type": "1-5"},
    {"key": "restlessness", "label": "Te simți agitat/ă sau neliniștit/ă în mod constant?", "type": "1-5"},
    {"key": "disorganization", "label": "Îți este greu să organizezi activitățile, spațiul sau documentele?",
     "type": "1-5"},
    {"key": "time_management_difficulty",
     "label": "Ai dificultăți în a gestiona timpul și a respecta termenele limită?", "type": "1-5"},
    {"key": "sadness_frequency", "label": "Te simți trist/ă sau gol/ă interior în mod frecvent?", "type": "1-5"},
    {"key": "loss_of_interest", "label": "Ai pierdut interesul pentru activitățile care îți plăceau înainte?",
     "type": "1-5"},
    {"key": "fatigue_level", "label": "Te simți obosit/ă constant, chiar și după odihnă?", "type": "1-5"},
    {"key": "hopelessness", "label": "Te simți fără speranță în legătură cu viitorul?", "type": "1-5"},
    {"key": "sleep_disturbance", "label": "Ai probleme cu somnul (adormit greu, trezit noaptea, dormit prea mult)?",
     "type": "1-5"},
    {"key": "appetite_change", "label": "Ai observat schimbări semnificative în poftă de mâncare sau în greutate?",
     "type": "1-5"},
    {"key": "worthlessness_feelings", "label": "Te simți lipsit/ă de valoare sau te învinovățești excesiv?",
     "type": "1-5"},
    {"key": "suicidal_ideation", "label": "Ai gânduri legate de a-ți face rău sau despre suicid?", "type": "1-5"},
    {"key": "worry_frequency", "label": "Îngrijorarea sau anxietatea îți afectează viața zilnică frecvent?",
     "type": "1-5"},
    {"key": "physical_tension",
     "label": "Simți tensiune fizică (mușchi încordați, dureri de cap, senzație de nod în stomac)?", "type": "1-5"},
    {"key": "panic_episodes", "label": "Ai episoade de panică (inimă accelerată, transpirație, senzație de pericol)?",
     "type": "1-5"},
    {"key": "avoidance_behavior", "label": "Eviți situații sau locuri din cauza anxietății sau fricii?", "type": "1-5"},
    {"key": "irritability", "label": "Simți iritabilitate sau furie frecvent, uneori fără motiv clar?", "type": "1-5"},
    {"key": "excessive_fear",
     "label": "Ai frici intense față de situații specifice (spații aglomerate, înălțimi etc.)?", "type": "1-5"},
    {"key": "difficulty_relaxing", "label": "Îți este greu să te relaxezi sau să-ți oprești gândurile?", "type": "1-5"},
    {"key": "intrusive_thoughts", "label": "Ai gânduri nedorite, deranjante, care îți revin în mod repetat?",
     "type": "1-5"},
    {"key": "compulsive_behaviors", "label": "Simți nevoia să efectuezi anumite acțiuni pentru a reduce anxietatea?",
     "type": "1-5"},
    {"key": "checking_behaviors",
     "label": "Verifici în mod repetat lucruri (ușa, aragazul, geanta) chiar și când știi că sunt în regulă?",
     "type": "1-5"},
    {"key": "contamination_fear", "label": "Te temi excesiv de murdărie, microbi sau contaminare?", "type": "1-5"},
    {"key": "symmetry_need",
     "label": "Simți nevoia ca lucrurile să fie perfect simetrice sau aranjate într-un anumit fel?", "type": "1-5"},
    {"key": "mental_rituals",
     "label": "Efectuezi ritualuri mentale (numărare, rugăciuni repetate) pentru a calma gândurile?", "type": "1-5"},
    {"key": "hallucinations", "label": "Auzi voci sau vezi lucruri pe care alții nu le percep?", "type": "1-5"},
    {"key": "delusions",
     "label": "Ai convingeri ferme despre lucruri pe care cei din jur le consideră false (ex: cineva te urmărește)?",
     "type": "1-5"},
    {"key": "disorganized_thinking",
     "label": "Simți că gândurile tale sunt confuze, dezorganizate sau greu de urmărit?", "type": "1-5"},
    {"key": "social_withdrawal", "label": "Te izolezi de familie și prieteni, evitând contactul social?",
     "type": "1-5"},
    {"key": "flat_affect", "label": "Ai dificultăți în a simți sau a exprima emoții (te simți amorțit/ă emoțional)?",
     "type": "1-5"},
    {"key": "cognitive_impairment", "label": "Ai probleme cu memoria, atenția sau luarea deciziilor?", "type": "1-5"},
    {"key": "paranoia", "label": "Simți că oamenii din jurul tău au intenții rele față de tine?", "type": "1-5"},
    {"key": "self_esteem_low", "label": "Stima ta de sine este scăzută sau fluctuantă?", "type": "1-5"},
]

labels = {
    'label_depression':    'Depresie',
    'label_anxiety':       'Anxietate',
    'label_adhd':          'ADHD',
    'label_asd':           'ASD (Autism)',
    'label_ocd':           'OCD',
    'label_schizophrenia': 'Schizofrenie',
}

labels_info = {
    "label_depression":    ("Depresie",     "https://ro.wikipedia.org/wiki/Depresie_(stare)"),
    "label_anxiety":       ("Anxietate",    "https://ro.wikipedia.org/wiki/Anxietate"),
    "label_adhd":          ("ADHD",         "https://ro.wikipedia.org/wiki/Tulburare_hiperchinetic%C4%83_cu_deficit_de_aten%C8%9Bie"),
    "label_asd":           ("ASD (Autism)", "https://ro.wikipedia.org/wiki/Autism"),
    "label_ocd":           ("OCD",          "https://ro.wikipedia.org/wiki/Tulburare_obsesiv-compulsiv%C4%83"),
    "label_schizophrenia": ("Schizofrenie", "https://ro.wikipedia.org/wiki/Schizofrenie"),
}

def predict(answers):
    row = {}
    for key, val in answers.items():
        if key == "gender":
            row[key] = map_gender.get(val, 0)
        elif key == "education_level":
            row[key] = map_education.get(val, 3)
        elif key == "employment_status":
            row[key] = map_job.get(val, 0)
        else:
            row[key] = val

    df_test = pd.DataFrame([row], columns=feature_cols)
    probs   = reg.predict_proba(df_test)
    return {key: probs[i][0][1] * 100 for i, key in enumerate(labels_info.keys())}

st.set_page_config(page_title="M.I.N.D.")

if "page" not in st.session_state:
    st.session_state.page = "intro"
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}

@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    reg = joblib.load(os.path.join(base, "model.joblib"))
    feature_cols = joblib.load(os.path.join(base, "feature.joblib"))
    label_cols = joblib.load(os.path.join(base, "label.joblib"))
    map_education = {'Fără studii': 1, 'Școală generală': 2, 'Liceu': 3, 'Studii superioare': 4, 'Doctorat/Master avansat': 5}
    map_gender = {'Female': 0, 'Male': 1, 'Other': 2}
    map_job = {'Retired': 0, 'Employed': 1, 'Student': 2, 'Unemployed': 3, 'Disabled': 4}
    return reg, feature_cols, label_cols, map_education, map_gender, map_job

reg, feature_cols, label_cols, map_education, map_gender, map_job = load_model()

def render_question(q):
    st.subheader(q["label"])
    key = "widget_" + q["key"]

    if q["type"] == "age":
        return st.number_input("Vârsta", min_value=1, max_value=120, value=18, key=key)

    elif q["type"] == "1-5":
        labels_scale = ["1 - Deloc", "2 - Rar", "3 - Uneori", "4 - Des", "5 - Întotdeauna"]
        choice = st.radio("", labels_scale, horizontal=True, key=key)
        return int(choice[0])

    elif q["type"] == "choice":
        return st.selectbox("", q["options"], key=key)

    elif q["type"] == "edu":
        options = list(edu_mapping.keys())
        return st.selectbox("", options, key=key)

    elif q["type"] == "yn":
        choice = st.radio("", ["Da", "Nu"], horizontal=True, key=key)
        return 1 if choice == "Da" else 0

#Intro
if st.session_state.page == "intro":
    st.title("M.I.N.D.")
    st.markdown("""
    Acest chestionar estimează probabilitatea unor tulburări mintale comune
    pe baza simptomelor tale. **Nu reprezintă un diagnostic medical.**
    """)
    st.warning("Dacă ai gânduri de suicid sau crize acute, contactează imediat un specialist.")

    if st.button("Începe testul", use_container_width=True):
        st.session_state.page = "questions"
        st.session_state.idx  = 0
        st.session_state.answers = {}
        st.rerun()

#Questions
elif st.session_state.page == "questions":
    idx   = st.session_state.idx
    total = len(questions)
    q = questions[idx]
    st.progress((idx) / total, text=f"Întrebarea {idx + 1} din {total}")
    st.markdown("---")
    value = render_question(q)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if idx > 0:
            if st.button("Înapoi", use_container_width=True):
                st.session_state.idx -= 1
                st.rerun()

    with col2:
        label = "Următoarea" if idx < total - 1 else "Vezi rezultatele"
        if st.button(label, use_container_width=True):
            st.session_state.answers[q["key"]] = value
            if idx < total - 1:
                st.session_state.idx += 1
            else:
                st.session_state.page = "results"
            st.rerun()
#Rezultat
elif st.session_state.page == "results":
    st.title("Rezultatele tale")
    st.markdown("---")

    results = predict(st.session_state.answers)
    high    = [(k, v) for k, v in results.items() if v >= 70]

    if high:
        st.warning("Probabilitate ridicată detectată pentru una sau mai multe tulburări. Consultați un specialist.")

    for label_key, prob in results.items():
        nume, wiki = labels_info[label_key]
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{nume}**")
            st.progress(float(prob) / 100)
            st.caption(f"{prob:.1f}%")
        with col2:
            if prob >= 70:
                st.link_button("Wikipedia", wiki)
        st.markdown("")

    st.markdown("---")
    st.info("Aceste rezultate nu reprezintă un diagnostic medical. Consultați un specialist pentru o evaluare profesională.")

    if st.button("Reia testul", use_container_width=True):
        st.session_state.page    = "intro"
        st.session_state.idx     = 0
        st.session_state.answers = {}
        st.rerun()