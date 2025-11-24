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
# 2. LLM 호출 함수들 (프로파일링, 생성, **수정**)
# -------------------------------------------------------------------------
def analyze_patient_profile(api_key, patient_info):
    client = openai.OpenAI(api_key=api_key)
    prompt = f"""
    당신은 임상 영양 전문가입니다. 환자 정보를 분석하여 '식단 설계 가이드라인' 3가지를 요약해주세요.
    [환자 정보] 나이:{patient_info['나이']}, 질환:당뇨({patient_info.get('당뇨병')})/고혈압({patient_info.get('고혈압')}), 연하장애:{patient_info.get('연하장애')}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "핵심만 간결하게 요약하세요."}, {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def generate_hybrid_meal_plan(api_key, patient_profile, guidelines, candidate_menus):
    client = openai.OpenAI(api_key=api_key)
    candidates_str = ""
    for cat, menus in candidate_menus.items():
        sample = random.sample(menus, min(len(menus), 15))
        candidates_str += f"- {cat}: {', '.join(sample)}\n"

    prompt = f"""
    [입력] 1.가이드라인:{guidelines} 2.후보메뉴:{candidates_str}
    [지시] 위 정보를 바탕으로 1끼 식단을 구성하세요. (밥,국,주찬,부찬,김치)
    [출력(JSON)] {{ "menu": {{...}}, "rationale": "..." }}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "JSON only."}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7
    )
    return json.loads(response.choices[0].message.content)

def refine_meal_plan(api_key, current_menu, feedback, candidate_menus):
    """
    [New] 전문가 피드백을 반영하여 식단을 수정하는 함수
    """
    client = openai.OpenAI(api_key=api_key)
    
    # 후보 메뉴 문자열 (선택지 제공)
    candidates_str = ""
    for cat, menus in candidate_menus.items():
        sample = random.sample(menus, min(len(menus), 15))
        candidates_str += f"- {cat}: {', '.join(sample)}\n"

    prompt = f"""
    [역할] 당신은 영양사의 피드백을 반영하여 식단을 수정하는 보조 AI입니다.
    
    [현재 식단]
    {json.dumps(current_menu, ensure_ascii=False)}
    
    [영양사 피드백 (수정 요청사항)]
    "{feedback}"
    
    [지시사항]
    1. 위 피드백을 반영하여 문제가 되는 메뉴를 **후보 메뉴 리스트** 내에서 적절한 것으로 교체하세요.
    2. 피드백과 관련 없는 메뉴는 그대로 유지하세요.
    3. 나트륨 저감 요청 시, 국물을 건더기 위주로 변경하거나 저염 메뉴를 선택하세요.
    
    [후보 메뉴 리스트]
    {candidates_str}
    
    [출력 형식 (JSON Only)]
    {{
        "menu": {{ "밥": "...", "국": "...", "주찬": "...", "부찬": "...", "김치": "..." }},
        "rationale": "수정된 이유 (피드백을 어떻게 반영했는지)"
    }}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "JSON only."}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7
    )
    return json.loads(response.choices[0].message.content)

# -------------------------------------------------------------------------
# 3. 영양 평가 검증 (Python Calculation)
# -------------------------------------------------------------------------
def validate_nutrition(generated_menu, nutrient_df):
    total_stats = {'에너지(kcal)': 0, '나트륨(mg)': 0, '단백질(g)': 0}
    validated_details = []
    
    for cat, menu_name in generated_menu.items():
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
            validated_details.append({'구분': cat, '메뉴': menu_name, 'Kcal': 0, 'Na': 0})
            
    return total_stats, pd.DataFrame(validated_details)

# -------------------------------------------------------------------------
# 4. 메인 UI
# -------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Hybrid Nutrition System (Interactive)")
    st.title("🔬 AI-Human 협업형 영양 급식 시스템")

    menu_df, nutrient_df, category_df, patient_df = load_data()
    if menu_df is None: return

    with st.sidebar:
        api_key = st.text_input("OpenAI API Key", type="password")
        selected_patient = st.selectbox("수급자 선택", patient_df['수급자명'].unique())

    # --- Step 1: 환자 프로파일 ---
    st.subheader(f"1. 환자 분석: {selected_patient}")
    patient_info = patient_df[patient_df['수급자명'] == selected_patient].iloc[0]
    
    if api_key and 'guidelines' not in st.session_state:
        with st.spinner("분석 중..."):
            st.session_state['guidelines'] = analyze_patient_profile(api_key, patient_info)
    
    if 'guidelines' in st.session_state:
        st.info(st.session_state['guidelines'])

    st.markdown("---")

    # --- Step 2: 식단 생성 (최초) ---
    st.subheader("2. 식단 생성 및 검증")
    
    # 후보 메뉴 준비 (공통)
    candidates = {}
    for cat in ['밥', '국', '주찬', '부찬', '김치']:
        candidates[cat] = category_df[category_df['Category'] == cat]['Menu'].unique().tolist()

    if st.button("🚀 초기 식단 생성"):
        if not api_key:
            st.error("API 키 필요")
        else:
            with st.spinner("생성 중..."):
                ai_result = generate_hybrid_meal_plan(api_key, patient_info, st.session_state['guidelines'], candidates)
                total_nutri, detail_df = validate_nutrition(ai_result['menu'], nutrient_df)
                
                st.session_state['generated_result'] = ai_result
                st.session_state['nutri_stats'] = total_nutri
                st.session_state['detail_df'] = detail_df

    # --- 결과 표시 및 Step 3 전문가 피드백 루프 ---
    if 'generated_result' in st.session_state:
        res = st.session_state['generated_result']
        stats = st.session_state['nutri_stats']
        df = st.session_state['detail_df']
        
        # AI 의도 표시
        st.success(f"🤖 **AI:** {res['rationale']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df, use_container_width=True)
        with col2:
            # 영양 검증 시각화
            target_kcal = float(patient_info['체중']) * 10
            st.metric("에너지(kcal)", f"{int(stats['에너지(kcal)'])}", delta=f"{int(stats['에너지(kcal)'] - target_kcal)}")
            
            na_val = int(stats['나트륨(mg)'])
            if pd.notna(patient_info.get('고혈압')) and na_val > 800:
                st.error(f"⚠️ 나트륨 {na_val}mg (고혈압 주의)")
            else:
                st.metric("나트륨(mg)", f"{na_val}")

        st.markdown("---")
        
        # === [핵심] Interactive Feedback Loop ===
        st.subheader("3. 전문가 검토 및 수정 (Interactive Feedback)")
        
        with st.form("feedback_loop"):
            feedback_text = st.text_input("수정 요청사항 입력", 
                                        placeholder="예: 국의 나트륨이 너무 높으니 다른 국으로 바꿔줘. 또는 부찬을 고기반찬으로 변경해줘.")
            
            c1, c2 = st.columns([1, 4])
            with c1:
                regen_btn = st.form_submit_button("🔄 피드백 반영하여 재생성")
            with c2:
                approve_btn = st.form_submit_button("✅ 최종 승인")
            
            if regen_btn and feedback_text:
                if not api_key:
                    st.error("API 키 확인 필요")
                else:
                    with st.spinner(f"AI가 '{feedback_text}' 내용을 반영하여 수정 중입니다..."):
                        # 1. 수정 함수 호출 (Refinement)
                        new_ai_result = refine_meal_plan(api_key, res['menu'], feedback_text, candidates)
                        
                        # 2. 다시 영양 검증 (Re-validation)
                        new_stats, new_df = validate_nutrition(new_ai_result['menu'], nutrient_df)
                        
                        # 3. 상태 업데이트 및 새로고침
                        st.session_state['generated_result'] = new_ai_result
                        st.session_state['nutri_stats'] = new_stats
                        st.session_state['detail_df'] = new_df
                        st.rerun() # 화면 즉시 갱신
            
            if approve_btn:
                st.balloons()
                st.success("식단이 최종 승인되었습니다! 조리실로 전송합니다.")
                st.json(res['menu'])

if __name__ == "__main__":
    main()
