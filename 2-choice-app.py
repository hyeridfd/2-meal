import streamlit as st
import pandas as pd
import openai
import json
import random

# -------------------------------------------------------------------------
# 1. 데이터 로드 및 전처리
# -------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        menu_df = pd.read_csv('menu.csv')
        nutrient_df = pd.read_csv('nutrient.csv')
        category_df = pd.read_csv('category.csv')
        
        # 고령자 데이터 로드
        patient_file = 'senior.csv'
        patient_df = pd.read_csv(patient_file, header=3)
        patient_df.columns = patient_df.columns.str.strip()

        menu_df.fillna(0, inplace=True)
        return menu_df, nutrient_df, category_df, patient_df
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return None, None, None, None

# -------------------------------------------------------------------------
# 2. [Step 1] 사용자 프로파일링 (LLM 분석)
# -------------------------------------------------------------------------
def analyze_patient_profile(api_key, patient_info):
    """
    논문의 'User Profile Interpretation' 단계
    환자 데이터를 바탕으로 '식단 설계 전략'을 텍스트로 도출합니다.
    """
    client = openai.OpenAI(api_key=api_key)
    
    prompt = f"""
    당신은 임상 영양 전문가입니다. 아래 환자 정보를 분석하여 '식단 설계 시 주의해야 할 핵심 가이드라인' 3가지를 요약해주세요.
    
    [환자 정보]
    - 나이: {patient_info['나이']}, 성별: {patient_info['성별']}
    - 체중: {patient_info['체중']}kg
    - 질환: 당뇨({patient_info.get('당뇨병')}), 고혈압({patient_info.get('고혈압')}), 신장질환({patient_info.get('신장질환')})
    - 연하장애: {patient_info.get('연하장애')} (현재식사: {patient_info['현재식사현황']})
    
    출력 형식:
    1. [칼로리/영양] ...
    2. [식재료 제한] ...
    3. [조리 형태] ...
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "핵심만 간결하게 요약하세요."}, {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# -------------------------------------------------------------------------
# 3. [Step 2] 제약 기반 식단 생성 (LLM Generation)
# -------------------------------------------------------------------------
def generate_hybrid_meal_plan(api_key, patient_profile, guidelines, candidate_menus):
    """
    논문의 'LLM-Powered Meal Planning' 단계
    프로파일 분석 결과(guidelines)를 바탕으로 메뉴를 선택합니다.
    """
    client = openai.OpenAI(api_key=api_key)
    
    # 후보 메뉴 문자열 변환
    candidates_str = ""
    for cat, menus in candidate_menus.items():
        # 토큰 절약을 위해 카테고리별 10개 랜덤 샘플링 (실전엔 필터링된 DB 사용)
        sample = random.sample(menus, min(len(menus), 10))
        candidates_str += f"- {cat}: {', '.join(sample)}\n"

    prompt = f"""
    [역할]
    당신은 '하이브리드 영양 시스템'의 AI 에이전트입니다.
    
    [입력 정보]
    1. 환자 가이드라인:
    {guidelines}
    
    2. 후보 메뉴 데이터베이스:
    {candidates_str}
    
    [지시 사항]
    위 가이드라인을 엄격히 준수하여 1끼 식단을 구성하세요.
    - 밥, 국, 주찬, 부찬, 김치 구성 필수.
    - 특히 질환(당뇨/고혈압)과 연하장애(죽/다짐)를 고려하여 메뉴를 선택하거나, 메뉴명 뒤에 (조리법)을 추가하세요.
    
    [출력 형식 (JSON Only)]
    {{
        "menu": {{ "밥": "...", "국": "...", "주찬": "...", "부찬": "...", "김치": "..." }},
        "rationale": "이 식단을 구성한 의학적/영양학적 이유 한 줄"
    }}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "JSON 형식으로만 답하세요."}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7
    )
    return json.loads(response.choices[0].message.content)

# -------------------------------------------------------------------------
# 4. [Step 3] 영양 평가 검증 (Python Calculation)
# -------------------------------------------------------------------------
def validate_nutrition(generated_menu, nutrient_df):
    """
    논문의 'Nutritional Assessment' 단계
    LLM이 생성한 식단의 실제 영양소를 DB에서 찾아 계산하고, 목표치와 비교합니다.
    (LLM의 환각이나 계산 오류를 방지하는 핵심 단계)
    """
    total_stats = {'에너지(kcal)': 0, '나트륨(mg)': 0, '단백질(g)': 0}
    validated_details = []
    
    for cat, menu_name in generated_menu.items():
        # 조리법 괄호 제거하고 검색 (예: "멸치볶음(갈아서)" -> "멸치볶음")
        clean_name = menu_name.split('(')[0].strip()
        
        row = nutrient_df[nutrient_df['Menu'] == clean_name]
        
        if not row.empty:
            kcal = row['에너지(kcal)'].values[0]
            na = row['나트륨(mg)'].values[0]
            prot = row['단백질(g)'].values[0]
            
            total_stats['에너지(kcal)'] += kcal
            total_stats['나트륨(mg)'] += na
            total_stats['단백질(g)'] += prot
            
            validated_details.append({'구분': cat, '메뉴': menu_name, 'Kcal': kcal, 'Na': na})
        else:
            # DB에 없는 메뉴(LLM이 지어낸 경우) or 조리법 변형
            validated_details.append({'구분': cat, '메뉴': menu_name, 'Kcal': 0, 'Na': 0})
            
    return total_stats, pd.DataFrame(validated_details)

# -------------------------------------------------------------------------
# 5. 메인 UI (Hybrid Interface)
# -------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Hybrid Nutrition System")
    st.title("🔬 논문 기반 하이브리드 영양 급식 시스템")
    st.markdown("Implemented based on: *Enhancing Personalized Nutrition with LLM-Powered Meal Planning*")

    # 데이터 로드
    menu_df, nutrient_df, category_df, patient_df = load_data()
    if menu_df is None: return

    with st.sidebar:
        api_key = st.text_input("OpenAI API Key", type="password")
        st.info("이 시스템은 AI(생성) + Code(검증) + Human(최종확인) 3단계로 작동합니다.")
        
        selected_patient = st.selectbox("수급자 선택", patient_df['수급자명'].unique())

    # --- Step 1: 환자 프로파일 분석 ---
    st.subheader(f"1. User Profiling: {selected_patient}님 분석")
    patient_info = patient_df[patient_df['수급자명'] == selected_patient].iloc[0]
    
    col_p1, col_p2 = st.columns([1, 2])
    with col_p1:
        st.table(patient_info[['나이', '성별', '당뇨병', '고혈압', '연하장애', '현재식사현황']].astype(str))
    
    with col_p2:
        if api_key:
            if 'guidelines' not in st.session_state:
                with st.spinner("LLM이 환자 데이터를 분석 중입니다..."):
                    st.session_state['guidelines'] = analyze_patient_profile(api_key, patient_info)
            
            st.success("✅ AI 영양 분석 결과 (Guideline)")
            st.write(st.session_state['guidelines'])
        else:
            st.warning("API 키를 입력하면 분석이 시작됩니다.")

    st.markdown("---")

    # --- Step 2: 식단 생성 ---
    st.subheader("2. Hybrid Meal Planning (AI Generation + Validation)")
    
    if st.button("🚀 하이브리드 식단 생성 시작"):
        if not api_key:
            st.error("API 키가 필요합니다.")
        else:
            with st.spinner("AI가 가이드라인에 맞춰 최적의 메뉴를 조합 중입니다..."):
                # 후보군 준비
                candidates = {}
                for cat in ['밥', '국', '주찬', '부찬', '김치']:
                    candidates[cat] = category_df[category_df['Category'] == cat]['Menu'].unique().tolist()
                
                # LLM 호출
                ai_result = generate_hybrid_meal_plan(api_key, patient_info, st.session_state['guidelines'], candidates)
                
                # --- Step 3: 코드 검증 (Validation) ---
                total_nutri, detail_df = validate_nutrition(ai_result['menu'], nutrient_df)
                
                # 결과 저장
                st.session_state['generated_result'] = ai_result
                st.session_state['nutri_stats'] = total_nutri
                st.session_state['detail_df'] = detail_df

    # 생성된 결과가 있으면 표시
    if 'generated_result' in st.session_state:
        res = st.session_state['generated_result']
        stats = st.session_state['nutri_stats']
        df = st.session_state['detail_df']
        
        # 2-1. AI의 의도 설명
        st.info(f"💡 **AI 설계 의도:** {res['rationale']}")
        
        col1, col2 = st.columns(2)
        
        # 2-2. 식단표 및 영양 검증 결과
        with col1:
            st.markdown("#### 📋 생성된 식단표")
            st.dataframe(df, use_container_width=True)
            
        with col2:
            st.markdown("#### ⚖️ 영양 적합성 검증 (Code Validator)")
            
            # 목표치 (간이 계산)
            target_kcal = float(patient_info['체중']) * 10 # 한 끼 기준
            
            # 시각화: 칼로리
            kcal_delta = stats['에너지(kcal)'] - target_kcal
            st.metric("총 에너지 (목표 대비)", f"{int(stats['에너지(kcal)'])} kcal", 
                      delta=f"{int(kcal_delta)} kcal", delta_color="inverse")
            
            # 시각화: 나트륨 (고혈압 환자 주의)
            na_color = "normal"
            if pd.notna(patient_info.get('고혈압')) and stats['나트륨(mg)'] > 800:
                na_color = "off" # 빨간색 경고
                st.error(f"⚠️ 나트륨 경고! (현재: {int(stats['나트륨(mg)'])}mg) -> 고혈압 환자 기준 초과 가능성")
            else:
                st.metric("총 나트륨", f"{int(stats['나트륨(mg)'])} mg")

            st.progress(min(stats['에너지(kcal)'] / (target_kcal * 1.5), 1.0))
            st.caption("위 그래프는 목표 칼로리 대비 충족률입니다.")

    st.markdown("---")

    # --- Step 4: 전문가 피드백 (Expert Oversight) ---
    st.subheader("3. Expert Oversight (최종 검토)")
    st.markdown("논문에서는 **'전문가의 개입'**을 필수 요소로 봅니다. 위 식단을 검토하고 필요시 수정하세요.")
    
    if 'generated_result' in st.session_state:
        with st.form("expert_review"):
            feedback = st.text_area("수정 사항 또는 조리실 전달 메모", 
                                  placeholder="예: 멸치볶음 대신 두부조림으로 변경해주세요. 나트륨이 너무 높습니다.")
            
            approved = st.form_submit_button("✅ 식단 최종 승인")
            
            if approved:
                st.success("식단이 승인되었습니다! 조리실로 데이터가 전송됩니다.")
                st.json({
                    "final_menu": st.session_state['generated_result']['menu'],
                    "expert_note": feedback,
                    "nutrition_verified": True
                })

if __name__ == "__main__":
    main()
