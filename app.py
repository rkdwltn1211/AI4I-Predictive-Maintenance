import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ── 페이지 설정 ────────────────────────────────────────────────
st.set_page_config(
    page_title="설비 고장 예측 대시보드",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── 전역 CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    .main-title {
        font-size: 2rem; font-weight: 700; color: #ffffff;
        letter-spacing: -0.5px; margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 0.85rem; color: #6b7280;
        margin-bottom: 1.5rem; letter-spacing: 0.5px;
    }
    .kpi-card {
        background: #1a1d27; border: 1px solid #2d3142;
        border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 0.5rem;
    }
    .kpi-label {
        font-size: 0.72rem; color: #6b7280;
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.4rem;
    }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #ffffff; line-height: 1; }
    .kpi-sub   { font-size: 0.75rem; color: #10b981; margin-top: 0.3rem; }
    .kpi-alert { color: #ef4444; }

    .section-header {
        font-size: 1rem; font-weight: 600; color: #ffffff;
        border-left: 3px solid #3b82f6;
        padding-left: 0.75rem; margin: 1.5rem 0 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1d27; border-radius: 8px;
        padding: 4px; gap: 4px; border: 1px solid #2d3142;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent; color: #9ca3af;
        border-radius: 6px; padding: 0.5rem 1.2rem;
        font-size: 0.85rem; font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important; color: #ffffff !important;
    }
    hr { border-color: #2d3142; }
    .info-box {
        background: #1a1d27; border: 1px solid #2d3142;
        border-radius: 8px; padding: 1rem 1.2rem;
        font-size: 0.85rem; color: #9ca3af; line-height: 1.8;
    }
    .info-box strong { color: #e0e0e0; }

    /* 위젯 글씨 */
    label, .stRadio label, .stCheckbox label {
        color: #ffffff !important;
        font-size: 0.85rem !important;
    }
    .stSelectbox div[data-baseweb="select"] span {
        color: #ffffff !important;
    }
    .stRadio div[role="radiogroup"] label p {
        color: #ffffff !important;
    }
    .stCheckbox span p,
    .stCheckbox label span,
    [data-testid="stCheckbox"] span,
    [data-testid="stCheckbox"] p {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Plotly 공통 테마 ───────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='#1a1d27',
    plot_bgcolor='#1a1d27',
    font=dict(color='#cbd5e1', size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor='#2d3142', linecolor='#2d3142', zerolinecolor='#2d3142'),
    yaxis=dict(gridcolor='#2d3142', linecolor='#2d3142', zerolinecolor='#2d3142'),
    legend=dict(
        font=dict(color='#ffffff'),
        bgcolor='#1a1d27',
        bordercolor='#2d3142',
        borderwidth=1
    ),
)
COLOR_NORMAL  = '#3b82f6'
COLOR_FAILURE = '#ef4444'
COLOR_SEQ     = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

# ── 데이터 로드 ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/ai4i2020.csv')
    df.columns = df.columns.str.strip()
    df['Temp_diff']    = df['Process temperature [K]'] - df['Air temperature [K]']
    df['Power']        = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * (2 * np.pi / 60)
    df['Torque_Wear']  = df['Torque [Nm]'] * df['Tool wear [min]']
    df['Type_encoded'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})
    df = df.rename(columns={'Tool wear [min]': 'Tool_wear_min'})
    df['상태'] = df['Machine failure'].map({0: '정상', 1: '고장'})
    return df

df = load_data()

# ── 헤더 ──────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏭 설비 고장 예측 대시보드</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">AI4I 2020 Predictive Maintenance &nbsp;|&nbsp; LightGBM &nbsp;|&nbsp; Threshold 0.70</div>', unsafe_allow_html=True)

# ── 탭 구성 ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA",
    "🔮 고장 예측",
    "🧠 SHAP 분석",
    "💰 비용 계산기"
])

# ════════════════════════════════════════════════════════════════
# 탭 1 — EDA
# ════════════════════════════════════════════════════════════════
with tab1:

    # ── KPI 카드 ───────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">전체 데이터</div>
            <div class="kpi-value">10,000</div>
            <div class="kpi-sub">건</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">정상 건수</div>
            <div class="kpi-value">9,661</div>
            <div class="kpi-sub">96.6%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">고장 건수</div>
            <div class="kpi-value kpi-alert">{df['Machine failure'].sum():,}</div>
            <div class="kpi-sub kpi-alert">3.4% — 불균형 데이터</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">사용 피처</div>
            <div class="kpi-value">5</div>
            <div class="kpi-sub">파생변수 3개 포함</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">최종 모델</div>
            <div class="kpi-value">LGBM</div>
            <div class="kpi-sub">Recall 0.85 · AUC 0.97</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── ① 수치형 변수 분포 ─────────────────────────────────────
    st.markdown('<div class="section-header">① 수치형 변수 분포</div>', unsafe_allow_html=True)

    col_options = {
        'Air temperature [K]'     : 'Air temperature [K]',
        'Process temperature [K]' : 'Process temperature [K]',
        'Rotational speed [rpm]'  : 'Rotational speed [rpm]',
        'Torque [Nm]'             : 'Torque [Nm]',
        'Tool_wear_min'           : 'Tool wear [min]',
        'Temp_diff'               : 'Temp_diff (파생)',
        'Power'                   : 'Power (파생)',
        'Torque_Wear'             : 'Torque_Wear (파생)',
    }

    left, right = st.columns([1, 3])
    with left:
        sel_col    = st.selectbox("변수 선택", list(col_options.keys()),
                                  format_func=lambda x: col_options[x])
        show_split = st.checkbox("고장/정상 분리", value=True)
        chart_type = st.radio("차트 유형", ["히스토그램", "Box Plot"], horizontal=True)

    with right:
        if chart_type == "히스토그램":
            if show_split:
                fig = px.histogram(
                    df, x=sel_col, color='상태',
                    color_discrete_map={'정상': COLOR_NORMAL, '고장': COLOR_FAILURE},
                    barmode='overlay', opacity=0.7, nbins=50,
                    title=f'{col_options[sel_col]} 분포'
                )
            else:
                fig = px.histogram(
                    df, x=sel_col, nbins=50,
                    color_discrete_sequence=[COLOR_NORMAL],
                    title=f'{col_options[sel_col]} 분포'
                )
        else:
            if show_split:
                fig = px.box(
                    df, x='상태', y=sel_col, color='상태',
                    color_discrete_map={'정상': COLOR_NORMAL, '고장': COLOR_FAILURE},
                    title=f'{col_options[sel_col]} — 고장/정상 비교'
                )
            else:
                fig = px.box(
                    df, y=sel_col,
                    color_discrete_sequence=[COLOR_NORMAL],
                    title=f'{col_options[sel_col]} Box Plot'
                )

        fig.update_layout(
            **PLOTLY_LAYOUT,
            title_font_color='#ffffff',
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── ② 고장 유형별 발생 현황 ────────────────────────────────
    st.markdown('<div class="section-header">② 고장 유형별 발생 현황</div>', unsafe_allow_html=True)

    failure_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    failure_ko   = {'TWF': '공구 마모', 'HDF': '열 방출', 'PWF': '전력', 'OSF': '과부하', 'RNF': '랜덤'}

    failure_counts = df[failure_cols].sum().reset_index()
    failure_counts.columns = ['Type', 'Count']
    failure_counts['Label'] = failure_counts['Type'].map(failure_ko)
    failure_counts = failure_counts.sort_values('Count', ascending=False)

    fa_left, fa_right = st.columns([3, 2])
    with fa_left:
        fig2 = go.Figure(go.Bar(
            x=failure_counts['Label'],
            y=failure_counts['Count'],
            marker_color=COLOR_SEQ,
            text=failure_counts['Count'],
            textposition='outside',
            textfont=dict(color='#e0e0e0', size=12),
        ))
        fig2.update_layout(
            **PLOTLY_LAYOUT,
            title_text='고장 유형별 발생 건수',
            title_font_color='#ffffff',
            yaxis_title='발생 건수',
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    with fa_right:
        st.markdown('<br>', unsafe_allow_html=True)
        summary = pd.DataFrame({
            '유형': ['HDF', 'OSF', 'PWF', 'TWF', 'RNF'],
            '건수': [115, 98, 95, 46, 19],
            '핵심 조건': [
                'Temp_diff ≤ 9K & rpm ≤ 1380',
                '토크 × 마모 임계값 초과',
                'Power < 3500 or > 9000W',
                'Tool wear ≥ 200분',
                '0.1% 랜덤 발생',
            ]
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

    st.divider()

    # ── ③ 고장 유형별 센서값 비교 ─────────────────────────────
    st.markdown('<div class="section-header">③ 고장 유형별 센서값 비교</div>', unsafe_allow_html=True)

    sensor_options = {
        'Temp_diff'              : 'Temp_diff',
        'Power'                  : 'Power',
        'Torque_Wear'            : 'Torque_Wear',
        'Tool_wear_min'          : 'Tool wear [min]',
        'Rotational speed [rpm]' : 'Rotational speed [rpm]',
        'Torque [Nm]'            : 'Torque [Nm]',
    }
    sel_sensor = st.selectbox("센서 선택", list(sensor_options.keys()),
                               format_func=lambda x: sensor_options[x], key='sensor_sel')

    rows = []
    for fc in failure_cols:
        avg = df[df[fc] == 1][sel_sensor].mean()
        rows.append({'유형': failure_ko[fc], '평균값': round(avg, 2)})
    rows.append({'유형': '정상', '평균값': round(df[df['Machine failure'] == 0][sel_sensor].mean(), 2)})
    compare_df = pd.DataFrame(rows)

    colors = COLOR_SEQ[:5] + [COLOR_NORMAL]
    fig3 = go.Figure(go.Bar(
        x=compare_df['유형'],
        y=compare_df['평균값'],
        marker_color=colors,
        text=compare_df['평균값'],
        textposition='outside',
        textfont=dict(color='#e0e0e0', size=11),
    ))
    fig3.update_layout(
        **PLOTLY_LAYOUT,
        title_text=f'고장 유형별 {sensor_options[sel_sensor]} 평균 비교',
        title_font_color='#ffffff',
        yaxis_title='평균값',
        showlegend=False
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── ④ 제품 등급(L/M/H)별 분석 ────────────────────────────
    st.markdown('<div class="section-header">④ 제품 등급 (L / M / H) 별 고장 분석</div>', unsafe_allow_html=True)

    grade_left, grade_right = st.columns(2)

    with grade_left:
        grade_stats = df.groupby('Type').agg(
            전체=('Machine failure', 'count'),
            고장=('Machine failure', 'sum')
        ).reset_index()
        grade_stats['고장률'] = (grade_stats['고장'] / grade_stats['전체'] * 100).round(2)

        fig4 = go.Figure(go.Bar(
            x=grade_stats['Type'],
            y=grade_stats['고장률'],
            marker_color=[COLOR_SEQ[0], COLOR_SEQ[2], COLOR_SEQ[3]],
            text=grade_stats['고장률'].astype(str) + '%',
            textposition='outside',
            textfont=dict(color='#e0e0e0'),
        ))
        fig4.update_layout(
            **PLOTLY_LAYOUT,
            title_text='제품 등급별 고장률 (%)',
            title_font_color='#ffffff',
            yaxis_title='고장률 (%)',
            showlegend=False
        )
        st.plotly_chart(fig4, use_container_width=True)

    with grade_right:
        sel_grade_sensor = st.selectbox(
            "센서 선택", list(sensor_options.keys()),
            format_func=lambda x: sensor_options[x], key='grade_sensor'
        )
        fig5 = px.violin(
            df, x='Type', y=sel_grade_sensor, color='Type',
            box=True, color_discrete_sequence=COLOR_SEQ,
            title=f'등급별 {sensor_options[sel_grade_sensor]} 분포'
        )
        fig5.update_layout(
            **PLOTLY_LAYOUT,
            title_font_color='#ffffff',
            showlegend=False
        )
        st.plotly_chart(fig5, use_container_width=True)

    st.divider()

    # ── ⑤ 상관관계 히트맵 ─────────────────────────────────────
    st.markdown('<div class="section-header">⑤ 피처 간 상관관계</div>', unsafe_allow_html=True)

    corr_cols = ['Air temperature [K]', 'Process temperature [K]',
                 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool_wear_min',
                 'Temp_diff', 'Power', 'Torque_Wear', 'Machine failure']
    corr = df[corr_cols].corr().round(3)

    fig6 = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate='%{text}',
        textfont=dict(size=10),
    ))
    fig6.update_layout(
        **PLOTLY_LAYOUT,
        title_text='피처 간 상관관계 히트맵',
        title_font_color='#ffffff',
        height=500
    )
    st.plotly_chart(fig6, use_container_width=True)

    with st.expander("💡 핵심 해석"):
        st.markdown("""
        <div class="info-box">
        <strong>Air temp ↔ Process temp: 0.88</strong> — 온도 자체보다 차이(Temp_diff)가 HDF 탐지에 핵심<br>
        <strong>Rotational speed ↔ Torque: -0.88</strong> — rpm↑이면 토크↓. Power 파생변수로 통합<br>
        <strong>Torque_Wear ↔ Tool wear: 0.90</strong> — 다중공선성 주의. 스케일링 후 VIF 확인 완료<br>
        <strong>rpm ↔ Machine failure: -0.044</strong> — 선형 상관 낮지만 이상치 구간에서 고장 비율 2.5배 → 비선형 관계
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 탭 2 — 고장 예측
# ════════════════════════════════════════════════════════════════
with tab2:
    import joblib
    import os

    st.header("🔮 실시간 고장 예측")
    st.caption("센서값을 입력하면 LightGBM 모델이 고장 확률을 실시간으로 예측합니다.")

    # ── 모델 로드 ──────────────────────────────────────────────
    @st.cache_resource
    def load_model():
        return joblib.load('lgbm_final.pkl')

    try:
        model = load_model()
        model_loaded = True
    except:
        model_loaded = False

    if not model_loaded:
        st.error("❌ lgbm_final.pkl 파일을 찾을 수 없어요. app.py와 같은 폴더에 있는지 확인해줘요.")
        st.stop()

    st.divider()

    # ── 입력 / 결과 레이아웃 ───────────────────────────────────
    inp, out = st.columns([1, 1])

    with inp:
        st.markdown('<div class="section-header">센서값 입력</div>', unsafe_allow_html=True)

        # 제품 등급
        grade = st.selectbox("제품 등급", ['L', 'M', 'H'],
                             help="L: 저가 50% / M: 중가 30% / H: 고가 20%")
        type_encoded = {'L': 0, 'M': 1, 'H': 2}[grade]

        # 슬라이더
        air_temp = st.slider(
            "Air Temperature [K]", 295.0, 305.0, 300.0, 0.1,
            help="공기 온도. 정상 범위: 295~305K"
        )
        proc_temp = st.slider(
            "Process Temperature [K]", 305.0, 315.0, 310.0, 0.1,
            help="공정 온도. Air temp보다 약 10K 높음"
        )
        rpm = st.slider(
            "Rotational Speed [rpm]", 1168, 2886, 1500,
            help="회전 속도. 1380 이하이면 HDF 위험"
        )
        torque = st.slider(
            "Torque [Nm]", 3.8, 76.6, 40.0, 0.1,
            help="토크. rpm과 음의 상관관계"
        )
        tool_wear = st.slider(
            "Tool Wear [min]", 0, 253, 100,
            help="공구 누적 사용시간. 200분 이상이면 TWF 위험"
        )

        # 파생변수 계산
        temp_diff   = proc_temp - air_temp
        power       = torque * rpm * (2 * np.pi / 60)
        torque_wear = torque * tool_wear

        # 파생변수 미리보기
        st.markdown('<div class="section-header">파생변수 (자동 계산)</div>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        d1.metric("Temp_diff", f"{temp_diff:.2f} K",
                  delta="⚠️ HDF 위험" if temp_diff < 8.6 else "정상",
                  delta_color="inverse" if temp_diff < 8.6 else "normal")
        d2.metric("Power", f"{power:.0f} W",
                  delta="⚠️ PWF 위험" if power < 3500 or power > 9000 else "정상",
                  delta_color="inverse" if power < 3500 or power > 9000 else "normal")
        d3.metric("Torque_Wear", f"{torque_wear:.0f}",
                  delta="⚠️ OSF 위험" if torque_wear > 11000 else "정상",
                  delta_color="inverse" if torque_wear > 11000 else "normal")

    with out:
        st.markdown('<div class="section-header">예측 결과</div>', unsafe_allow_html=True)

        # ── 예측 실행 ──────────────────────────────────────────
        features = pd.DataFrame([[type_encoded, temp_diff, power, torque_wear, tool_wear]],
                                 columns=['Type_encoded', 'Temp_diff', 'Power',
                                          'Torque_Wear', 'Tool_wear_min'])
        proba = model.predict_proba(features)[0][1]
        THRESHOLD = 0.70

        # ── 상태 판정 ──────────────────────────────────────────
        if proba >= THRESHOLD:
            status_color = '#ef4444'
            status_label = '⚠️ 고장 위험'
            status_bg    = '#2d1a1a'
        elif proba >= 0.4:
            status_color = '#f59e0b'
            status_label = '🔶 주의 필요'
            status_bg    = '#2d2510'
        else:
            status_color = '#10b981'
            status_label = '✅ 정상'
            status_bg    = '#0f2d1e'

        # ── 고장 확률 게이지 ───────────────────────────────────
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(proba * 100, 1),
            number={'suffix': '%', 'font': {'color': '#ffffff', 'size': 40}},
            gauge={
                'axis': {'range': [0, 100],
                         'tickcolor': '#cbd5e1',
                         'tickfont': {'color': '#cbd5e1'}},
                'bar': {'color': status_color},
                'bgcolor': '#2d3142',
                'steps': [
                    {'range': [0, 40],   'color': '#0f2d1e'},
                    {'range': [40, 70],  'color': '#2d2510'},
                    {'range': [70, 100], 'color': '#2d1a1a'},
                ],
                'threshold': {
                    'line': {'color': '#ffffff', 'width': 3},
                    'thickness': 0.85,
                    'value': THRESHOLD * 100
                }
            },
            title={'text': '고장 확률', 'font': {'color': '#ffffff', 'size': 16}}
        ))
        fig_gauge.update_layout(
            paper_bgcolor='#1a1d27',
            font=dict(color='#cbd5e1'),
            height=280,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── 상태 배지 ──────────────────────────────────────────
        st.markdown(f"""
        <div style="background:{status_bg}; border:1px solid {status_color};
                    border-radius:10px; padding:1rem 1.5rem; text-align:center;
                    margin-bottom:1rem;">
            <div style="font-size:1.4rem; font-weight:700;
                        color:{status_color};">{status_label}</div>
            <div style="font-size:0.85rem; color:#9ca3af; margin-top:0.3rem;">
                고장 확률 {proba*100:.1f}% | Threshold {THRESHOLD*100:.0f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 고장 유형 추정 ─────────────────────────────────────
        st.markdown('<div class="section-header">고장 유형 추정</div>', unsafe_allow_html=True)

        failure_risks = []

        if temp_diff < 8.6 and rpm < 1380:
            failure_risks.append(('HDF', '열 방출 실패', '#ef4444',
                                  f'Temp_diff={temp_diff:.2f}K (기준 <8.6K), rpm={rpm} (기준 <1380)'))
        if power < 3500 or power > 9000:
            failure_risks.append(('PWF', '전력 실패', '#f59e0b',
                                  f'Power={power:.0f}W (정상 범위: 3500~9000W)'))
        osf_thresh = {'L': 11000, 'M': 12000, 'H': 13000}[grade]
        if torque_wear > osf_thresh:
            failure_risks.append(('OSF', '과부하 실패', '#8b5cf6',
                                  f'Torque_Wear={torque_wear:.0f} (기준 >{osf_thresh})'))
        if tool_wear >= 200:
            failure_risks.append(('TWF', '공구 마모 실패', '#3b82f6',
                                  f'Tool wear={tool_wear}분 (기준 ≥200분)'))

        if failure_risks:
            for code, name, color, reason in failure_risks:
                st.markdown(f"""
                <div style="background:#1a1d27; border-left:3px solid {color};
                            border-radius:6px; padding:0.7rem 1rem; margin-bottom:0.5rem;">
                    <div style="color:{color}; font-weight:600; font-size:0.9rem;">
                        {code} — {name}
                    </div>
                    <div style="color:#9ca3af; font-size:0.8rem; margin-top:0.2rem;">
                        {reason}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#0f2d1e; border:1px solid #10b981;
                        border-radius:8px; padding:1rem; text-align:center;
                        color:#10b981; font-size:0.9rem;">
                고장 조건에 해당하는 유형 없음 — 현재 센서값은 정상 범위입니다
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 탭 3 — SHAP 분석
# ════════════════════════════════════════════════════════════════
with tab3:
    import shap

    st.header("🧠 SHAP 분석")
    st.caption("모델이 예측할 때 각 피처가 얼마나, 어떤 방향으로 영향을 미쳤는지 설명합니다.")

    FEATURES = ['Type_encoded', 'Temp_diff', 'Power', 'Torque_Wear', 'Tool_wear_min']
    FEATURE_LABELS = {
        'Type_encoded' : 'Type (등급)',
        'Temp_diff'    : 'Temp_diff',
        'Power'        : 'Power',
        'Torque_Wear'  : 'Torque_Wear',
        'Tool_wear_min': 'Tool wear [min]',
    }

    # ── 테스트 데이터 준비 ─────────────────────────────────────
    @st.cache_data
    def get_test_data():
        from sklearn.model_selection import train_test_split
        X = df[FEATURES]
        y = df['Machine failure']
        _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return X_test

    # ── SHAP 값 계산 (캐싱) ────────────────────────────────────
    @st.cache_data
    def get_shap_values(_model, X_test):
        explainer  = shap.TreeExplainer(_model)
        shap_vals  = explainer.shap_values(X_test)
        # 버전에 따라 list 또는 array
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        return shap_vals, explainer.expected_value

    X_test = get_test_data()

    with st.spinner("SHAP 값 계산 중... 처음 한 번만 시간이 걸려요 ⏳"):
        try:
            model_shap = load_model()
            shap_vals, expected_val = get_shap_values(model_shap, X_test)
            shap_loaded = True
        except Exception as e:
            st.error(f"SHAP 계산 실패: {e}")
            shap_loaded = False

    if shap_loaded:

        st.divider()

        # ── ① 피처 중요도 Bar Plot ─────────────────────────────
        st.markdown('<div class="section-header">① 피처 중요도 (SHAP 평균)</div>', unsafe_allow_html=True)

        mean_shap = np.abs(shap_vals).mean(axis=0)
        importance_df = pd.DataFrame({
            'Feature' : [FEATURE_LABELS[f] for f in FEATURES],
            'SHAP'    : mean_shap
        }).sort_values('SHAP', ascending=True)

        fig_imp = go.Figure(go.Bar(
            x=importance_df['SHAP'],
            y=importance_df['Feature'],
            orientation='h',
            marker_color=COLOR_SEQ[:len(FEATURES)],
            text=importance_df['SHAP'].round(3),
            textposition='outside',
            textfont=dict(color='#e0e0e0', size=11),
        ))
        fig_imp.update_layout(
            **PLOTLY_LAYOUT,
            title_text='피처별 평균 |SHAP| 값 (고장 클래스 기준)',
            title_font_color='#ffffff',
            xaxis_title='평균 |SHAP| 값',
            showlegend=False,
            height=320
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        with st.expander("💡 해석 보기"):
            st.markdown("""
            <div class="info-box">
            <strong>SHAP 값이 클수록</strong> 해당 피처가 예측에 더 큰 영향을 미친다는 뜻이에요.<br>
            절댓값 기준이라 방향(올림/내림)은 아래 Summary Plot에서 확인할 수 있어요.<br><br>
            <strong>EDA 상관계수와 다를 수 있어요</strong> — 상관계수는 선형 관계만 측정하지만,
            SHAP은 비선형 관계까지 잡아내기 때문이에요.
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── ② Summary Plot (Plotly로 재현) ────────────────────
        st.markdown('<div class="section-header">② Summary Plot — 피처값과 영향 방향</div>', unsafe_allow_html=True)

        # 샘플 수 제한 (속도)
        n_sample = min(500, len(X_test))
        idx      = np.random.choice(len(X_test), n_sample, replace=False)
        X_sample = X_test.iloc[idx].reset_index(drop=True)
        S_sample = shap_vals[idx]

        fig_sum = go.Figure()
        for i, feat in enumerate(FEATURES):
            feat_vals  = X_sample[feat].values
            shap_col   = S_sample[:, i]
            # 피처값을 0~1로 정규화해서 색상에 매핑
            v_min, v_max = feat_vals.min(), feat_vals.max()
            norm = (feat_vals - v_min) / (v_max - v_min + 1e-9)


            fig_sum.add_trace(go.Scatter(
                x=shap_col,
                y=[i] * n_sample,
                mode='markers',
                marker=dict(
                    size=7,
                    color=norm,
                    colorscale='RdBu_r',
                    opacity=0.75,
                    line=dict(width=0),
                    showscale=(i == 0),
                    colorbar=dict(
                        title=dict(
                            text='피처값<br>(낮음→높음)',
                            font=dict(color='#ffffff')
                        ),
                        tickfont=dict(color='#cbd5e1'),
                        x=1.02
                    )
                ),
                name=FEATURE_LABELS[feat],
                showlegend=False,
                hovertemplate=f'<b>{FEATURE_LABELS[feat]}</b><br>피처값: %{{text}}<br>SHAP: %{{x:.3f}}<extra></extra>',
                text=[f'{v:.2f}' for v in feat_vals],
            ))

        # y축 눈금을 피처 이름으로
        fig_sum.update_yaxes(
            tickvals=list(range(len(FEATURES))),
            ticktext=[FEATURE_LABELS[f] for f in FEATURES],
            tickfont=dict(color='#cbd5e1', size=12),
        )

        fig_sum.update_layout(
            **PLOTLY_LAYOUT,
            title_text='SHAP Summary Plot (파란색=피처값 낮음, 빨간색=피처값 높음)',
            title_font_color='#ffffff',
            xaxis_title='SHAP 값 (고장 확률에 미치는 영향)',
            height=380,
        )
        fig_sum.add_vline(x=0, line_color='#6b7280', line_dash='dash', line_width=1)
        st.plotly_chart(fig_sum, use_container_width=True)

        with st.expander("💡 해석 보기"):
            st.markdown("""
            <div class="info-box">
            <strong>SHAP 양수(오른쪽)</strong> → 고장 확률을 높이는 방향<br>
            <strong>SHAP 음수(왼쪽)</strong> → 고장 확률을 낮추는 방향<br><br>
            <strong>Temp_diff — 파란색(낮은 값)이 오른쪽</strong> → 온도차가 낮을수록 고장 확률 올라감 (HDF)<br>
            <strong>Tool wear — 빨간색(높은 값)이 오른쪽</strong> → 마모 높을수록 고장 확률 올라감 (TWF)<br>
            <strong>Power — 빨간색(높은 값)이 오른쪽</strong> → 전력 높을수록 고장 확률 올라감 (PWF)
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── ③ Waterfall Plot — 탭 2 입력값 기준 ──────────────
        st.markdown('<div class="section-header">③ Waterfall Plot — 현재 입력값 기준 예측 설명</div>', unsafe_allow_html=True)
        st.caption("탭 2에서 설정한 센서값을 기준으로 모델이 왜 이렇게 예측했는지 설명해요.")

        # 탭 2 입력값 가져오기 (session_state 없으면 기본값)
        try:
            grade_enc = {'L': 0, 'M': 1, 'H': 2}.get(grade, 0)
            td  = temp_diff
            pw  = power
            tw  = torque_wear
            twm = tool_wear
        except:
            grade_enc, td, pw, tw, twm = 1, 10.0, 5000.0, 4000.0, 100

        sample = pd.DataFrame([[grade_enc, td, pw, tw, twm]], columns=FEATURES)

        explainer_single = shap.TreeExplainer(model_shap)
        sv_single = explainer_single.shap_values(sample)
        if isinstance(sv_single, list):
            sv_single = sv_single[1]
        sv_single = sv_single[0]

        if isinstance(expected_val, (list, np.ndarray)):
            base_val = float(expected_val[1])
        else:
            base_val = float(expected_val)

        feat_names = [FEATURE_LABELS[f] for f in FEATURES]
        feat_values = sample.iloc[0].values

        # ── 진짜 Waterfall Chart ──────────────────────────────
        # 영향 작은 순서로 정렬 (아래→위로 쌓이므로 역순)
        sorted_idx   = np.argsort(np.abs(sv_single))
        sorted_names = [f"{feat_names[i]}={feat_values[i]:.1f}" for i in sorted_idx]
        sorted_shap  = [sv_single[i] for i in sorted_idx]

        # measure: 'relative'=누적 막대, 'total'=합계 막대
        # 기준값 먼저, 피처들, 최종값 순서
        measures = ['absolute'] + ['relative'] * len(sorted_shap)
        y_labels = ['기준값 E[f(X)]'] + sorted_names
        values   = [base_val] + sorted_shap

        # 색상: 양수=빨강, 음수=파랑, total=회색
        decreasing_color = COLOR_NORMAL   # 파랑
        increasing_color = COLOR_FAILURE  # 빨강

        fig_wf = go.Figure(go.Waterfall(
            orientation='h',
            measure=measures,
            y=y_labels,
            x=values,
            base=base_val,
            decreasing=dict(marker=dict(color=COLOR_NORMAL)),
            increasing=dict(marker=dict(color=COLOR_FAILURE)),
            totals=dict(marker=dict(color='#6b7280')),
            textposition='outside',
            text=[f'{base_val:.3f}'] + [f'{v:+.3f}' for v in sorted_shap],
            textfont=dict(color='#e0e0e0', size=11),
            connector=dict(line=dict(color='#4b5563', width=1, dash='dot')),
        ))
        fig_wf.update_layout(
            **PLOTLY_LAYOUT,
            title_text=f'Waterfall Plot  |  기준값: {base_val:.3f}  →  최종 예측: {base_val + sv_single.sum():.3f}',
            title_font_color='#ffffff',
            xaxis_title='SHAP 누적값',
            height=380,
            showlegend=False,
        )
        fig_wf.add_vline(x=0, line_color='#6b7280', line_dash='dash', line_width=1)
        st.plotly_chart(fig_wf, use_container_width=True)

        with st.expander("💡 해석 보기"):
            st.markdown("""
            <div class="info-box">
            <strong>빨간 막대</strong> → 고장 확률을 높이는 피처<br>
            <strong>파란 막대</strong> → 고장 확률을 낮추는 피처<br>
            <strong>막대 길이</strong> → 영향력 크기<br><br>
            탭 2에서 슬라이더를 바꾸면 이 차트도 자동으로 업데이트돼요.
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 탭 4 — 비용 계산기
# ════════════════════════════════════════════════════════════════
with tab4:
    st.header("💰 비용 절감 계산기")
    st.caption("모델 도입 전후 비용을 비교하고 ROI를 계산합니다.")

    st.divider()

    inp4, out4 = st.columns([1, 1])

    with inp4:
        st.markdown('<div class="section-header">파라미터 설정</div>', unsafe_allow_html=True)

        n_machines   = st.slider("공장 가동 기계 대수", 100, 2000, 500, 50)
        failure_rate = st.slider("월 고장 발생 비율 (%)", 1.0, 10.0, 3.4, 0.1) / 100
        repair_cost  = st.slider("고장 1건당 수리비 (만원)", 100, 2000, 500, 50)
        downtime     = st.slider("고장 1건당 생산중단 손실 (만원)", 100, 3000, 800, 100)
        inspect_cost = st.slider("예방 점검 1회당 비용 (만원)", 10, 200, 50, 10)

        st.markdown('<div class="section-header">모델 성능</div>', unsafe_allow_html=True)
        recall       = st.slider("모델 Recall (고장 탐지율)", 0.5, 1.0, 0.85, 0.01)
        fpr          = st.slider("오탐률 (정상→고장 오분류)", 0.0, 0.2, 0.03, 0.01)

    with out4:
        st.markdown('<div class="section-header">월간 비용 분석</div>', unsafe_allow_html=True)

        # ── 계산 ───────────────────────────────────────────────
        monthly_failures = round(n_machines * failure_rate)
        total_normal     = n_machines - monthly_failures

        # 모델 도입 전
        cost_before = monthly_failures * (repair_cost + downtime)

        # 모델 도입 후
        detected     = round(monthly_failures * recall)        # 탐지된 고장
        missed       = monthly_failures - detected             # 놓친 고장
        false_alarms = round(total_normal * fpr)               # 오탐 (불필요한 점검)

        cost_inspect    = detected * inspect_cost              # 탐지된 고장 → 예방 점검
        cost_missed     = missed * (repair_cost + downtime)    # 놓친 고장 → 사후 수리
        cost_false      = false_alarms * inspect_cost          # 오탐 → 불필요한 점검
        cost_after      = cost_inspect + cost_missed + cost_false

        saved_monthly   = cost_before - cost_after
        saved_yearly    = saved_monthly * 12
        saving_rate     = saved_monthly / cost_before * 100 if cost_before > 0 else 0

        # ── KPI 카드 ───────────────────────────────────────────
        k1, k2 = st.columns(2)
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">도입 전 월 손실</div>
                <div class="kpi-value kpi-alert">{cost_before:,}</div>
                <div class="kpi-sub kpi-alert">만원/월</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">도입 후 월 손실</div>
                <div class="kpi-value">{cost_after:,}</div>
                <div class="kpi-sub">만원/월</div>
            </div>""", unsafe_allow_html=True)

        k3, k4 = st.columns(2)
        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">월 절감액</div>
                <div class="kpi-value" style="color:#10b981">{saved_monthly:,}</div>
                <div class="kpi-sub">절감률 {saving_rate:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">연간 절감액</div>
                <div class="kpi-value" style="color:#10b981">{saved_yearly:,}</div>
                <div class="kpi-sub">만원/년</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── 비용 구성 비교 차트 ────────────────────────────────────
    st.markdown('<div class="section-header">도입 전/후 비용 구성 비교</div>', unsafe_allow_html=True)

    ch1, ch2 = st.columns(2)

    with ch1:
        # 도입 전/후 총비용 비교
        fig_bar = go.Figure(go.Bar(
            x=['도입 전', '도입 후'],
            y=[cost_before, cost_after],
            marker_color=[COLOR_FAILURE, COLOR_NORMAL],
            text=[f'{cost_before:,}만원', f'{cost_after:,}만원'],
            textposition='outside',
            textfont=dict(color='#e0e0e0', size=12),
        ))
        fig_bar.update_layout(
            **PLOTLY_LAYOUT,
            title_text='월 총 손실 비교',
            title_font_color='#ffffff',
            yaxis_title='비용 (만원)',
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with ch2:
        # 도입 후 비용 구성 파이차트
        fig_pie = go.Figure(go.Pie(
            labels=['예방 점검비 (탐지)', '사후 수리비 (미탐지)', '불필요한 점검비 (오탐)'],
            values=[cost_inspect, cost_missed, cost_false],
            marker_colors=[COLOR_NORMAL, COLOR_FAILURE, '#f59e0b'],
            textfont=dict(color='#ffffff', size=11),
            hole=0.4,
        ))
        fig_pie.update_layout(
            **PLOTLY_LAYOUT,
            title_text='도입 후 비용 구성',
            title_font_color='#ffffff',
            
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── 월별 누적 절감액 시뮬레이션 ───────────────────────────
    st.markdown('<div class="section-header">연간 누적 절감액 시뮬레이션</div>', unsafe_allow_html=True)

    months       = list(range(1, 13))
    cumulative   = [saved_monthly * m for m in months]

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=months,
        y=cumulative,
        mode='lines+markers',
        line=dict(color=COLOR_NORMAL, width=2),
        marker=dict(size=7, color=COLOR_NORMAL),
        fill='tozeroy',
        fillcolor='rgba(59,130,246,0.15)',
        name='누적 절감액',
        hovertemplate='%{x}개월: %{y:,}만원<extra></extra>'
    ))
    fig_line.add_hline(
        y=saved_yearly, line_dash='dash',
        line_color='#10b981', line_width=1,
        annotation_text=f'연간 절감액 {saved_yearly:,}만원',
        annotation_font_color='#10b981'
    )
    fig_line.update_layout(
        **PLOTLY_LAYOUT,
        title_text='월별 누적 절감액',
        title_font_color='#ffffff',
        xaxis_title='경과 월',
        yaxis_title='누적 절감액 (만원)',
        
        showlegend=False,
        height=320
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.divider()

    # ── 상세 내역 테이블 ───────────────────────────────────────
    st.markdown('<div class="section-header">상세 내역</div>', unsafe_allow_html=True)

    detail = pd.DataFrame({
        '항목': [
            '월 고장 건수',
            '탐지된 고장 (예방 점검)',
            '놓친 고장 (사후 수리)',
            '오탐 (불필요한 점검)',
            '도입 전 월 손실',
            '도입 후 월 손실',
            '월 절감액',
            '절감률',
            '연간 절감액',
        ],
        '값': [
            f'{monthly_failures}건',
            f'{detected}건 (Recall {recall*100:.0f}%)',
            f'{missed}건',
            f'{false_alarms}건',
            f'{cost_before:,}만원',
            f'{cost_after:,}만원',
            f'{saved_monthly:,}만원',
            f'{saving_rate:.1f}%',
            f'{saved_yearly:,}만원',
        ]
    })
    st.dataframe(detail, hide_index=True, use_container_width=True)
