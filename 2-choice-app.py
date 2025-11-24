import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import openai
import os

# -------------------------------------------------------------------------
# 1. 설정 및 데이터 로드
# -------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="AI 요양원 맞춤 급식 시스템")

@st.cache_data
def load_data():
    try:
        # 파일명은 실제 파일명과 정확히 일치해야 합니다.
        menu_df = pd.read_csv('menu.csv')
        nutrient_df = pd.read_csv('nutrient.csv')
        category_df = pd.read_csv('category.csv')
        ingredient_df = pd.read_csv('ingredient.csv')
        
        # 고령자 데이터는 상단 4줄이 헤더가 아니므로 skiprows=4 옵션 사용
        patient_df = pd.read_csv('senior.csv', header=4)
        
        # 데이터 전처리
        menu_df.fillna(0, inplace=True)
        return menu_df, nutrient_df, category_df, ingredient_df, patient_df
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None, None, None, None, None

# -------------------------------------------------------------------------
# 2. LLM 호출 함수 (OpenAI GPT)
# -------------------------------------------------------------------------
def ask_gpt_recipe(api_key, patient_info, menu_name, ingredients, nutrient_info):
    """
    LLM에게 환자 정보와 재료를 주고 맞춤형 레시피 수정을 요청하는 함수
    """
    if not api_key:
        return "⚠️ API 키가 입력되지 않아 AI 레시피를 생성할 수 없습니다. (테스트 모드)"

    client = openai.OpenAI(api_key=api_key)
    
    # 프롬프트 설계 (LLM에게 역할을 부여)
    system_prompt = "당신은 요양원 전문 임상 영양사입니다. 환자의 질환과 연하(삼킴) 능력을 고려하여 안전하고 영양가 있는 조리법을 수정해 주세요."
    
    user_prompt = f"""
    [환자 정보]
    - 나이: {patient_info['나이']}
    - 질환: {', '.join([k for k, v in patient_info.items() if k in ['당뇨병', '고혈압'] and pd.notna(v)])}
    - 연하장애 여부: {'있음' if pd.notna(patient_info['연하장애']) else '없음'}
    - 현재 식사 형태: {patient_info['현재식사현황']}

    [메뉴 정보]
    - 메뉴명: {menu_name}
    - 기존 재료: {ingredients}
    - 기본 영양: 에너지 {nutrient_info['에너지(kcal)']}kcal, 나트륨 {nutrient_info['나트륨(mg)']}mg

    [요청 사항]
    위 환자가 이 메뉴를 안전하게 섭취할 수 있도록 구체적인 '조리 지침'과 '재료 변경 사항'을 작성해 주세요.
    특히 나트륨 조절과 식감(다짐/갈기 등)에 집중해서 설명해 주세요.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o", # 또는 gpt-3.5-turbo
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM 호출 중 오류 발생: {e}"

# -------------------------------------------------------------------------
# 3. 메인 로직 및 UI
# -------------------------------------------------------------------------
def main():
    st.title("🏥 AI 기반 요양원 개인 맞춤형 급식 시스템")
    
    # 사이드바: 설정 및 입력
    with st.sidebar:
        st.header("⚙️ 설정")
        api_key = st.text_input("OpenAI API Key", type="password", help="키가 없으면 LLM 기능은 작동하지 않습니다.")
        
        menu_df, nutrient_df, category_df, ingredient_df, patient_df = load_data()
        if menu_df is None: return

        # 날짜 및 환자 선택
        selected_date = st.selectbox("📅 날짜 선택", menu_df.columns[1:])
        selected_patient_name = st.selectbox("🧓 수급자(환자) 선택", patient_df['수급자명'].dropna().unique())

    # --- 환자 정보 로드 ---
    patient_info = patient_df[patient_df['수급자명'] == selected_patient_name].iloc[0]
    
    # 환자 프로필 표시
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**수급자명:** {patient_info['수급자명']} ({patient_info['나이']}세/{patient_info['성별']})")
    with col2:
        conditions = []
        if pd.notna(patient_info['당뇨병']): conditions.append("당뇨")
        if pd.notna(patient_info['고혈압']): conditions.append("고혈압")
        if pd.notna(patient_info['연하장애']): conditions.append("연하장애 ⚠️")
        st.warning(f"**주요 질환:** {', '.join(conditions) if conditions else '없음'}")
    with col3:
        st.success(f"**식사 형태:** {patient_info['현재식사현황']}")

    st.markdown("---")

    # --- 식단 데이터 가져오기 (해당 날짜의 조식 6개 메뉴 가정) ---
    raw_menu_list = menu_df[selected_date].dropna().head(6).values

    st.subheader(f"🍛 {selected_date} 맞춤 식단 분석")

    # 2열 레이아웃: 메뉴 리스트 | 상세 AI 분석 결과
    left_col, right_col = st.columns([1, 1.5])

    selected_menu_for_ai = None
    
    with left_col:
        st.markdown("### 오늘의 메뉴 리스트")
        for menu in raw_menu_list:
            # 영양 정보 및 재료 가져오기
            nutri = nutrient_df[nutrient_df['Menu'] == menu]
            cat = category_df[category_df['Menu'] == menu]['Category'].values[0] if not category_df[category_df['Menu'] == menu].empty else "기타"
            
            # 위험 요소 감지 (규칙 기반)
            warning_tags = []
            if pd.notna(patient_info['연하장애']) and cat not in ['국', '죽']:
                warning_tags.append("🔴 식감주의")
            if pd.notna(patient_info['고혈압']) and not nutri.empty and nutri['나트륨(mg)'].values[0] > 600:
                warning_tags.append("🟠 나트륨주의")
            
            # 카드 형태로 표시
            with st.expander(f"**{cat}: {menu}** {' '.join(warning_tags)}"):
                if not nutri.empty:
                    st.write(f"- 칼로리: {nutri['에너지(kcal)'].values[0]} kcal")
                    st.write(f"- 나트륨: {nutri['나트륨(mg)'].values[0]} mg")
                
                # '이 메뉴 AI 분석하기' 버튼
                if st.button(f"🤖 {menu} AI 레시피 생성", key=menu):
                    selected_menu_for_ai = menu

    # --- AI 레시피 생성 영역 ---
    with right_col:
        st.markdown("### 🤖 AI 영양사 조리 지침")
        
        if selected_menu_for_ai:
            st.info(f"선택된 메뉴: **{selected_menu_for_ai}** 분석 중...")
            
            # 1. DB에서 재료 정보 긁어오기 (RAG)
            ingredients_rows = ingredient_df[ingredient_df['Menu'] == selected_menu_for_ai]
            ingredients_str = ", ".join(ingredients_rows['Ingredient'].unique())
            
            # 2. DB에서 영양 정보 가져오기
            nutri_info = nutrient_df[nutrient_df['Menu'] == selected_menu_for_ai].iloc[0]
            
            # 3. LLM 호출
            with st.spinner("AI가 환자 상태에 맞는 레시피를 작성하고 있습니다..."):
                ai_recipe = ask_gpt_recipe(api_key, patient_info, selected_menu_for_ai, ingredients_str, nutri_info)
            
            # 4. 결과 출력
            st.markdown(ai_recipe)
            
        else:
            st.write("👈 왼쪽 메뉴에서 [AI 레시피 생성] 버튼을 눌러보세요.")
            st.write("환자의 질환(당뇨, 연하장애 등)과 보유한 레시피 데이터를 결합하여 맞춤형 조리법을 제안합니다.")
            
            # 차트 예시 (전체 영양)
            st.markdown("#### 📊 식단 영양 요약")
            total_na = 0
            for m in raw_menu_list:
                n = nutrient_df[nutrient_df['Menu'] == m]
                if not n.empty: total_na += n['나트륨(mg)'].values[0]
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = total_na,
                title = {'text': "총 나트륨 (mg)"},
                gauge = {'axis': {'range': [None, 3000]},
                         'bar': {'color': "red" if total_na > 2000 else "green"},
                         'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 2000}}
            ))
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
