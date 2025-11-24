import streamlit as st
import pandas as pd
import openai
import json
import random

# -------------------------------------------------------------------------
# 1. 데이터 로드
# -------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # 파일 로드
        menu_df = pd.read_csv('menu.csv')
        nutrient_df = pd.read_csv('nutrient.csv')
        category_df = pd.read_csv('category.csv')
        ingredient_df = pd.read_csv('ingredient.csv')
        
        # 고령자 데이터 로드 (헤더 자동 찾기)
        patient_file = 'senior.csv'
        patient_df = pd.read_csv(patient_file, header=3)
        patient_df.columns = patient_df.columns.str.strip()
        return menu_df, nutrient_df, category_df, ingredient_df, patient_df
    except Exception as e:
        st.error(f"데이터 로드 중 오류: {e}")
        return None, None, None, None, None

# -------------------------------------------------------------------------
# 2. [핵심] LLM 식단 설계 에이전트
# -------------------------------------------------------------------------
def generate_meal_plan_by_llm(api_key, patient_info, inventory_list, candidate_menus):
    """
    LLM에게 '환자정보 + 재고 + 후보메뉴'를 주고 식단을 직접 짜오게 시키는 함수
    """
    if not api_key:
        return None

    client = openai.OpenAI(api_key=api_key)

    # 1. 프롬프트 데이터 준비
    # 너무 많은 데이터를 보내면 토큰이 터지므로, 카테고리별로 후보를 추려서 보냅니다.
    candidates_str = ""
    for cat, menus in candidate_menus.items():
        # 카테고리별 랜덤 10개씩만 후보로 줘서 선택하게 함 (실제론 DB 필터링 후 전달)
        sample = random.sample(menus, min(len(menus), 15)) 
        candidates_str += f"- {cat}: {', '.join(sample)}\n"

    # 2. 시스템 프롬프트 (AI의 역할 정의)
    system_role = """
    당신은 요양원 수석 영양사입니다. 
    제공된 [환자 정보]와 [보유 재고]를 고려하여, [후보 메뉴 리스트] 중에서 가장 적합한 1끼 식단을 구성하세요.
    
    [필수 규칙]
    1. 구성: 밥, 국, 주찬, 부찬, 김치 (총 5가지)
    2. 환자의 질환(당뇨, 고혈압)과 연하장애(씹는 능력)를 최우선으로 고려할 것.
    3. 가능한 [보유 재고]에 포함된 재료를 사용하는 메뉴를 우선 선택할 것.
    4. 선택한 메뉴가 환자에게 부적합할 경우(예: 연하장애인데 딱딱한 반찬), 메뉴 이름 뒤에 조리법 수정사항을 괄호로 적을 것. (예: 멸치볶음(갈아서 제공))
    
    [출력 형식]
    반드시 아래 JSON 형식으로만 답변하세요. 다른 말은 하지 마세요.
    {
        "reasoning": "왜 이 식단을 짰는지에 대한 3줄 요약 설명",
        "menu": {
            "밥": "메뉴명",
            "국": "메뉴명",
            "주찬": "메뉴명",
            "부찬": "메뉴명",
            "김치": "메뉴명"
        }
    }
    """

    # 3. 유저 프롬프트 (이번 건)
    user_prompt = f"""
    [환자 정보]
    - 나이/성별: {patient_info['나이']}세 / {patient_info['성별']}
    - 질환: 당뇨({patient_info.get('당뇨병')}), 고혈압({patient_info.get('고혈압')})
    - 연하장애: {patient_info.get('연하장애', '없음')}
    - 현재 식사 형태: {patient_info['현재식사현황']}

    [보유 재고 (많음)]
    {', '.join(inventory_list)}

    [후보 메뉴 리스트 (이 중에서 골라)]
    {candidates_str}
    """

    # 4. LLM 호출
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # gpt-3.5-turbo보다 gpt-4o가 JSON을 훨씬 잘 짭니다.
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}, # 강제로 JSON만 뱉게 설정
            temperature=0.7
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"LLM 호출 에러: {e}")
        return None

# -------------------------------------------------------------------------
# 3. 메인 UI
# -------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="AI 주도형 식단 설계")
    st.title("🧠 LLM 주도형 요양원 식단 생성기")
    st.markdown("규칙이 아닌, **AI의 판단**으로 환자 상태와 재고에 맞춰 식단을 짭니다.")

    # 1. 설정 및 데이터
    with st.sidebar:
        api_key = st.text_input("OpenAI API Key", type="password")
        if not api_key: st.warning("키를 입력해야 AI가 작동합니다.")
        menu_df, nutrient_df, category_df, ingredient_df, patient_df = load_data()

    if menu_df is None: return

    # 2. 화면 구성
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1. 대상 선택")
        selected_patient = st.selectbox("수급자 선택", patient_df['수급자명'].unique())
        patient_info = patient_df[patient_df['수급자명'] == selected_patient].iloc[0]

        # 환자 상태 카드
        st.info(f"""
        **{patient_info['수급자명']}** 님
        - 🩸 질환: 당뇨({patient_info.get('당뇨병','X')}), 고혈압({patient_info.get('고혈압','X')})
        - 🦷 연하: {patient_info.get('연하장애','없음')} ({patient_info['현재식사현황']})
        """)

        st.subheader("2. 가상 재고 설정")
        # 데모를 위해 재고 상황을 랜덤으로 가정
        all_ingredients = ingredient_df['Ingredient'].unique().tolist()
        # 매번 바뀌는 재고 상황 시뮬레이션
        if 'today_inventory' not in st.session_state:
            st.session_state['today_inventory'] = random.sample(all_ingredients, 20)
        
        inventory_list = st.session_state['today_inventory']
        st.write("📦 **오늘의 풍부한 식자재:**")
        st.write(", ".join(inventory_list[:10]) + " 등...")
        
        if st.button("🎲 재고 상황 바꾸기"):
            st.session_state['today_inventory'] = random.sample(all_ingredients, 20)
            st.rerun()

    with col2:
        st.subheader("3. AI 식단 설계 결과")
        
        if st.button("🚀 LLM에게 식단 설계 지시하기", type="primary"):
            if not api_key:
                st.error("API 키가 필요합니다.")
            else:
                with st.spinner("AI 영양사가 환자 정보와 냉장고를 확인하고 있습니다..."):
                    # 1. 후보 메뉴 리스트 준비 (DB에서 카테고리별로 분류)
                    candidates = {}
                    for cat in ['밥', '국', '주찬', '부찬', '김치']:
                        candidates[cat] = category_df[category_df['Category'] == cat]['Menu'].unique().tolist()

                    # 2. LLM 호출
                    ai_result = generate_meal_plan_by_llm(api_key, patient_info, inventory_list, candidates)

                    if ai_result:
                        # 결과 출력
                        st.success("식단 설계 완료!")
                        
                        # 1. AI의 생각 (Reasoning)
                        st.markdown(f"### 💡 AI의 설계 의도\n> {ai_result['reasoning']}")
                        
                        # 2. 식단표 시각화
                        menu_plan = ai_result['menu']
                        
                        # 영양 정보 매핑 (선택된 메뉴의 영양소 가져오기)
                        total_kcal = 0
                        total_na = 0
                        
                        plan_display = []
                        for cat, menu_name in menu_plan.items():
                            # 괄호(조리법) 제거하고 DB 매칭 시도
                            clean_name = menu_name.split('(')[0].strip()
                            
                            # 영양소 찾기
                            nutri = nutrient_df[nutrient_df['Menu'] == clean_name]
                            kcal = nutri['에너지(kcal)'].values[0] if not nutri.empty else 0
                            na = nutri['나트륨(mg)'].values[0] if not nutri.empty else 0
                            
                            total_kcal += kcal
                            total_na += na
                            
                            plan_display.append({
                                "구분": cat,
                                "AI 추천 메뉴": menu_name, # 조리법 포함된 이름
                                "칼로리(kcal)": round(kcal, 1),
                                "나트륨(mg)": round(na, 1)
                            })
                            
                        st.table(pd.DataFrame(plan_display))
                        
                        # 3. 영양 요약 차트
                        st.markdown("#### 📊 영양 분석")
                        col_a, col_b = st.columns(2)
                        col_a.metric("총 칼로리", f"{int(total_kcal)} kcal")
                        col_b.metric("총 나트륨", f"{int(total_na)} mg", 
                                     delta="주의" if total_na > 2000 else "적정", 
                                     delta_color="inverse")
                        
                    else:
                        st.error("식단을 생성하지 못했습니다. 다시 시도해주세요.")

if __name__ == "__main__":
    main()
