import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Tying the Data Knot",
    page_icon="💘",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }
.hero-title { font-family: 'DM Serif Display', serif; font-size: 2.6rem; line-height: 1.1; margin: 0; }
.hero-sub   { font-size: 1rem; color: gray; margin-top: 0.4rem; font-weight: 300; }
.metric-card { background: white; border-radius: 14px; padding: 1.25rem;
               border: 1px solid #f0e8e0; text-align: center; }
.metric-value { font-family: 'DM Serif Display', serif; font-size: 2rem;
                color: #c94f7c; margin: 0; }
.metric-label { font-size: 0.75rem; color: #999; text-transform: uppercase;
                letter-spacing: .07em; margin-top: 4px; }
.section-header { font-family: 'DM Serif Display', serif; font-size: 1.4rem;
                  border-bottom: 2px solid #c94f7c; padding-bottom: 6px; margin-bottom: 1rem; }
.result-serious { background: #d4edda; border: 1px solid #b8dfc4;
                  border-radius: 12px; padding: 1.25rem; text-align: center; }
.result-casual  { background: #ffeeba; border: 1px solid #f5c518;
                  border-radius: 12px; padding: 1.25rem; text-align: center; }
.result-title   { font-family: 'DM Serif Display', serif; font-size: 1.6rem; margin: 0; }
</style>
""", unsafe_allow_html=True)


# ── Load data & train RF ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model — please wait...")
def load_model():
    df = pd.read_csv('dating_app_behavior_dataset.csv')

    if 'interest_tags' in df.columns:
        df['interest_count'] = df['interest_tags'].apply(
            lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)

    core_features = [
        'bio_length', 'likes_received', 'app_usage_time_min',
        'message_sent_count', 'emoji_usage_rate', 'swipe_right_ratio',
        'mutual_matches', 'profile_pics_count', 'sexual_orientation'
    ]

    df_m = df[core_features + ['match_outcome']].copy()

    for col in df_m.columns:
        if df_m[col].dtype == 'object' or str(df_m[col].dtype) in ['string', 'StringDtype']:
            df_m[col] = df_m[col].fillna(df_m[col].mode()[0])
        elif pd.api.types.is_numeric_dtype(df_m[col]):
            df_m[col] = df_m[col].fillna(df_m[col].median())
        else:
            df_m[col] = df_m[col].fillna(df_m[col].mode()[0])

    orientation_stats = df_m.groupby('sexual_orientation').agg({
        'bio_length'        : 'mean',
        'message_sent_count': 'mean',
        'emoji_usage_rate'  : 'mean',
        'swipe_right_ratio' : 'mean'
    }).to_dict('index')

    def determine_seriousness(row):
        score = 0
        s = orientation_stats[row['sexual_orientation']]
        if (s['bio_length'] - 100) <= row['bio_length'] <= (s['bio_length'] + 100): score += 1
        if row['swipe_right_ratio'] < s['swipe_right_ratio']: score += 1
        if row['message_sent_count'] > s['message_sent_count']: score += 1
        if (s['emoji_usage_rate'] - 0.15) <= row['emoji_usage_rate'] <= (s['emoji_usage_rate'] + 0.15): score += 1
        if row['app_usage_time_min'] > 0 and (row['message_sent_count'] / row['app_usage_time_min']) > 0.5: score += 1
        if row['swipe_right_ratio'] > 0 and (row['mutual_matches'] / row['swipe_right_ratio']) > 5: score += 1
        if row['match_outcome'] in ['No Action', 'Blocked']: score -= 1
        return 1 if score >= 4 else 0

    df_m['is_serious'] = df_m.apply(determine_seriousness, axis=1)

    X = df_m.drop(columns=['is_serious', 'match_outcome'])
    y = df_m['is_serious']

    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose       import ColumnTransformer
    from sklearn.impute        import SimpleImputer
    from sklearn.pipeline      import Pipeline

    numeric_features     = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = ['sexual_orientation']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler',  StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    X_proc = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    rf.fit(X_train, y_train)

    return rf, preprocessor, df_m, orientation_stats


rf_model, preprocessor, df_model, orientation_stats = load_model()

serious_rate = df_model['is_serious'].mean()
total_users  = len(df_model)

# ── Real scores from notebook ─────────────────────────────────────────────────
leaderboard_data = pd.DataFrame([
    {'Model': 'Random Forest',       'Accuracy': 0.9055, 'F1-Score': 0.8631},
    {'Model': 'Neural Network',      'Accuracy': 0.8980, 'F1-Score': 0.8511},
    {'Model': 'Decision Tree',       'Accuracy': 0.8732, 'F1-Score': 0.8052},
    {'Model': 'KNN',                 'Accuracy': 0.8115, 'F1-Score': 0.7116},
    {'Model': 'Logistic Regression', 'Accuracy': 0.7722, 'F1-Score': 0.6251},
    {'Model': 'SVM',                 'Accuracy': 0.7721, 'F1-Score': 0.6249},
])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💘 Tying the Data Knot")
    st.markdown("*Love, Life & Likes*")
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Overview",
        "🔮 Live Predictor",
        "📊 Model Comparison",
        "🏳️‍🌈 LGBTQ+ Analysis",
        "ℹ️ About"
    ])
    st.divider()
    st.markdown("<small>Group 15 · WIA1006/WID3006</small>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("<h1 class='hero-title'>Tying the Data Knot 💘</h1>",
                unsafe_allow_html=True)
    st.markdown("<p class='hero-sub'>Predicting serious relationship intent across LGBTQ+ communities</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <p class='metric-value'>{total_users:,}</p>
            <p class='metric-label'>Total users</p></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <p class='metric-value'>{serious_rate*100:.1f}%</p>
            <p class='metric-label'>Serious daters</p></div>""",
            unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <p class='metric-value'>90.6%</p>
            <p class='metric-label'>RF Accuracy</p></div>""",
            unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <p class='metric-value'>0.863</p>
            <p class='metric-label'>RF F1-Score</p></div>""",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<p class='section-header'>Serious rate by LGBTQ+ group</p>",
                    unsafe_allow_html=True)
        grp    = df_model.groupby('sexual_orientation')['is_serious'] \
                         .mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        colors  = plt.cm.RdPu(np.linspace(0.35, 0.9, len(grp)))
        bars    = ax.barh(grp.index, grp.values * 100, color=colors)
        ax.set_xlabel('Serious rate (%)')
        ax.set_xlim(0, 65)
        for bar, val in zip(bars, grp.values):
            ax.text(bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height()/2,
                    f'{val*100:.1f}%', va='center', fontsize=9)
        ax.set_title('Serious dater rate per orientation', fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("<p class='section-header'>Casual vs serious split</p>",
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        counts  = df_model['is_serious'].value_counts()
        ax.pie(counts, labels=['Casual', 'Serious'],
               autopct='%1.1f%%', startangle=90,
               colors=['#f0c4d4', '#c94f7c'],
               wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax.set_title('Overall target distribution', fontsize=11)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>How is_serious is defined</p>",
                unsafe_allow_html=True)
    st.info("""
**The Goldilocks Rule** — A user scores 1 point per rule (needs ≥ 4 out of 6 to be Serious):

✅ Bio length within ±100 of their orientation group average  
✅ Swipe right ratio below their group average (selective)  
✅ Message count above their group average (engaged)  
✅ Emoji usage within ±0.15 of their group average  
✅ Message density > 0.5 messages per minute  
✅ Match ROI > 5 (selective swiping that leads to real matches)  
⛔ Penalty if outcome is Blocked or No Action
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>Dataset overview</p>",
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total records",  "50,000")
        st.metric("Total features", "19")
    with col2:
        orientations = df_model['sexual_orientation'].nunique()
        st.metric("LGBTQ+ groups", orientations)
        st.metric("Match outcomes", df_model['match_outcome'].nunique()
                  if 'match_outcome' in df_model.columns else "10")
    with col3:
        st.metric("Serious users",
                  f"{df_model['is_serious'].sum():,}")
        st.metric("Casual users",
                  f"{(df_model['is_serious']==0).sum():,}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Live Predictor":
    st.markdown("<h1 class='hero-title'>Live Predictor 🔮</h1>",
                unsafe_allow_html=True)
    st.markdown("<p class='hero-sub'>Enter a user's profile — Random Forest predicts their relationship intent in real time</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    orientations = sorted(df_model['sexual_orientation'].unique().tolist())
    col1, col2   = st.columns([1, 1])

    with col1:
        st.markdown("#### 👤 Profile Details")
        orientation  = st.selectbox("Sexual Orientation", orientations)
        stats        = orientation_stats[orientation]
        bio_length   = st.slider("Bio Length (characters)", 0, 500,
                                 int(stats['bio_length']))
        profile_pics = st.slider("Profile Pictures", 0, 6, 3)
        emoji_usage  = st.slider("Emoji Usage Rate", 0.0, 1.0,
                                 round(float(stats['emoji_usage_rate']), 2), 0.01)

        st.markdown("#### 📱 App Behaviour")
        app_usage    = st.slider("App Usage (min/day)", 1, 300, 60)
        swipe_ratio  = st.slider("Swipe Right Ratio", 0.0, 1.0,
                                 round(float(stats['swipe_right_ratio']), 2), 0.01)
        msg_count    = st.slider("Messages Sent", 0, 100,
                                 int(stats['message_sent_count']))
        likes        = st.slider("Likes Received", 0, 200, 100)
        matches      = st.slider("Mutual Matches", 0, 50, 10)
        predict_btn  = st.button("✨ Predict Relationship Intent")

    with col2:
        st.markdown("#### 📋 Prediction Result")

        if predict_btn:
            input_df = pd.DataFrame([{
                'bio_length'         : bio_length,
                'likes_received'     : likes,
                'app_usage_time_min' : app_usage,
                'message_sent_count' : msg_count,
                'emoji_usage_rate'   : emoji_usage,
                'swipe_right_ratio'  : swipe_ratio,
                'mutual_matches'     : matches,
                'profile_pics_count' : profile_pics,
                'sexual_orientation' : orientation
            }])

            X_input = preprocessor.transform(input_df)
            pred    = rf_model.predict(X_input)[0]

            s     = orientation_stats[orientation]
            rules = {
                'Bio length in range'   : (s['bio_length']-100) <= bio_length <= (s['bio_length']+100),
                'Selective swiper'      : swipe_ratio < s['swipe_right_ratio'],
                'Active messenger'      : msg_count > s['message_sent_count'],
                'Natural emoji usage'   : (s['emoji_usage_rate']-0.15) <= emoji_usage <= (s['emoji_usage_rate']+0.15),
                'High message density'  : (msg_count/app_usage > 0.5) if app_usage > 0 else False,
                'Good match ROI'        : (matches/swipe_ratio > 5) if swipe_ratio > 0 else False,
            }
            score = sum(rules.values())

            if pred == 1:
                st.markdown(f"""<div class='result-serious'>
                    <p class='result-title'>💚 Serious Dater</p>
                    <p style='margin:4px 0 0;color:#555'>Score: {score}/6 rules met</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class='result-casual'>
                    <p class='result-title'>💛 Casual Dater</p>
                    <p style='margin:4px 0 0;color:#555'>Score: {score}/6 rules met</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>**Score Breakdown:**", unsafe_allow_html=True)
            for rule, passed in rules.items():
                st.markdown(f"{'✅' if passed else '❌'} {rule}")

            st.markdown("<br>**Group averages used as thresholds:**",
                        unsafe_allow_html=True)
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.metric("Avg bio length",   f"{s['bio_length']:.0f} chars")
                st.metric("Avg swipe ratio",  f"{s['swipe_right_ratio']:.2f}")
            with col_g2:
                st.metric("Avg messages",     f"{s['message_sent_count']:.0f}")
                st.metric("Avg emoji rate",   f"{s['emoji_usage_rate']:.2f}")
        else:
            st.info("👈 Fill in the sliders and click **Predict** to see the result.")
            st.markdown("""
**How it works:**
The Random Forest model was trained on 50,000 dating app users.
It uses 8 behavioural features and compares each user against
their own LGBTQ+ orientation group averages — not a one-size-fits-all threshold.

The score breakdown shows exactly which of the 6 Goldilocks
rules the user meets or fails.
            """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.markdown("<h1 class='hero-title'>Model Comparison 📊</h1>",
                unsafe_allow_html=True)
    st.markdown("<p class='hero-sub'>Performance of all 6 models — Random Forest selected as best</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<p class='section-header'>🏆 Model Leaderboard</p>",
                unsafe_allow_html=True)

    def highlight_best(row):
        if row['Model'] == 'Random Forest':
            return ['background-color: #f0faf5'] * len(row)
        return [''] * len(row)

    display_df = leaderboard_data.copy()
    display_df['Accuracy'] = display_df['Accuracy'].apply(lambda x: f'{x*100:.2f}%')
    display_df['F1-Score'] = display_df['F1-Score'].apply(lambda x: f'{x:.4f}')
    display_df.index = range(1, len(display_df)+1)
    display_df.index.name = 'Rank'
    st.dataframe(display_df.style.apply(highlight_best, axis=1),
                 use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>Accuracy vs F1-Score</p>",
                unsafe_allow_html=True)

    fig, ax  = plt.subplots(figsize=(12, 5))
    x        = np.arange(len(leaderboard_data))
    width    = 0.35
    acc_vals = leaderboard_data['Accuracy'].values
    f1_vals  = leaderboard_data['F1-Score'].values
    names    = leaderboard_data['Model'].values

    bars1 = ax.bar(x - width/2, acc_vals * 100, width,
                   label='Accuracy (%)', color='#f0c4d4', edgecolor='white')
    bars2 = ax.bar(x + width/2, f1_vals * 100,  width,
                   label='F1-Score (×100)', color='#c94f7c', edgecolor='white')

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{bar.get_height():.1f}',
                ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Score')
    ax.set_title('All 6 models — Accuracy vs F1-Score', fontsize=12)
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>Why Random Forest?</p>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.success("**Highest F1-Score**  \n0.8631 — best balance of precision and recall")
    with c2:
        st.success("**Highest Accuracy**  \n90.55% — correctly classifies 9 in 10 users")
    with c3:
        st.success("**Robust to imbalance**  \nEnsemble of trees handles minority class well")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>F1-Score vs Accuracy — what's the difference?</p>",
                unsafe_allow_html=True)
    st.info("""
**Accuracy** tells you how many predictions were correct overall.
It can be misleading when classes are imbalanced — a model that predicts
everyone as "casual" still gets ~62% accuracy.

**F1-Score** balances precision (how many predicted serious are actually serious)
and recall (how many actual serious users were found).
F1 is the primary metric here because we care about correctly identifying serious daters,
not just getting the overall count right.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LGBTQ+ ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏳️‍🌈 LGBTQ+ Analysis":
    st.markdown("<h1 class='hero-title'>LGBTQ+ Analysis 🏳️‍🌈</h1>",
                unsafe_allow_html=True)
    st.markdown("<p class='hero-sub'>How serious relationship intent varies across sexual orientation communities</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    orientations = sorted(df_model['sexual_orientation'].unique())
    col1, col2   = st.columns(2)

    with col1:
        st.markdown("<p class='section-header'>Serious rate per group</p>",
                    unsafe_allow_html=True)
        grp_rate = df_model.groupby('sexual_orientation')['is_serious'] \
                           .mean().sort_values(ascending=False)
        fig, ax  = plt.subplots(figsize=(7, 5))
        colors   = plt.cm.RdPu(np.linspace(0.3, 0.9, len(grp_rate)))
        bars     = ax.bar(grp_rate.index, grp_rate.values * 100,
                          color=colors, edgecolor='white', width=0.6)
        for bar, val in zip(bars, grp_rate.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f'{val*100:.1f}%', ha='center', fontsize=9)
        ax.set_ylabel('Serious rate (%)')
        ax.set_ylim(0, 80)
        ax.tick_params(axis='x', rotation=30)
        ax.set_title('% of serious daters per orientation', fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("<p class='section-header'>Group behaviour averages</p>",
                    unsafe_allow_html=True)
        stats_df = pd.DataFrame(orientation_stats).T.round(2)
        stats_df.columns = ['Avg Bio', 'Avg Messages', 'Avg Emoji', 'Avg Swipe']
        st.dataframe(stats_df, use_container_width=True)
        st.caption("""
These averages are used as personalised thresholds in the Goldilocks scoring.
Each user is compared against their own community — not a global standard.
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>Casual vs serious split by orientation</p>",
                unsafe_allow_html=True)
    dist = df_model.groupby('sexual_orientation')['is_serious'] \
                   .value_counts(normalize=True).unstack().fillna(0) * 100
    dist.columns = ['Casual', 'Serious']
    fig, ax = plt.subplots(figsize=(12, 5))
    dist[['Casual', 'Serious']].plot(
        kind='bar', stacked=True, ax=ax,
        color=['#f0c4d4', '#c94f7c'],
        edgecolor='white', width=0.6)
    ax.set_ylabel('Proportion (%)')
    ax.set_xlabel('Sexual Orientation')
    ax.set_title('Casual vs serious breakdown per LGBTQ+ group', fontsize=12)
    ax.tick_params(axis='x', rotation=30)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 120)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>Explore a specific group</p>",
                unsafe_allow_html=True)
    selected = st.selectbox("Select orientation group", orientations)
    grp_df   = df_model[df_model['sexual_orientation'] == selected]

    g1, g2, g3, g4 = st.columns(4)
    with g1: st.metric("Total users",    f"{len(grp_df):,}")
    with g2: st.metric("Serious daters", f"{grp_df['is_serious'].sum():,}")
    with g3: st.metric("Casual daters",  f"{(grp_df['is_serious']==0).sum():,}")
    with g4: st.metric("Serious rate",   f"{grp_df['is_serious'].mean()*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, col, label in [
        (axes[0], 'message_sent_count', 'Messages sent'),
        (axes[1], 'swipe_right_ratio',  'Swipe right ratio'),
        (axes[2], 'bio_length',         'Bio length')
    ]:
        ax.hist(grp_df[grp_df['is_serious']==1][col],
                alpha=0.7, color='#c94f7c', label='Serious', bins=20)
        ax.hist(grp_df[grp_df['is_serious']==0][col],
                alpha=0.7, color='#f0c4d4', label='Casual',  bins=20)
        ax.set_title(f'{selected} — {label}', fontsize=10)
        ax.set_xlabel(label)
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f'Behaviour distribution — {selected}', fontsize=12, y=1.02)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("<h1 class='hero-title'>About This Project ℹ️</h1>",
                unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        st.markdown("""
### WIA1006/WID3006 Machine Learning
**Group 15**

---

### Problem Statement
*Can we predict whether a user on a dating app is a serious dater,
and does this prediction differ across LGBTQ+ sexual orientation groups?*

---

### Target Variable — `is_serious`
An engineered label using the **Goldilocks Rule** (score ≥ 4 out of 6 criteria):
- Bio length within ±100 of orientation group average
- Swipe ratio below group average (selective)
- Message count above group average (engaged)
- Emoji usage within ±0.15 of group average
- Message density > 0.5 msgs/min
- Match ROI > 5

---

### Dataset
Dating App Behavior Dataset — 50,000 synthetic records · 19 features  
[Kaggle Dataset](https://www.kaggle.com/datasets/keyushnisar/dating-app-behavior-dataset)
        """)

    with col2:
        st.markdown("### Model Results")
        st.dataframe(leaderboard_data.style.apply(
            lambda row: ['background-color: #f0faf5']*len(row)
            if row['Model'] == 'Random Forest' else ['']*len(row), axis=1
        ), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Features Used")
        features = ['bio_length', 'message_sent_count', 'swipe_right_ratio',
                    'emoji_usage_rate', 'app_usage_time_min', 'mutual_matches',
                    'likes_received', 'profile_pics_count', 'sexual_orientation']
        for f in features:
            st.markdown(f"- `{f}`")
