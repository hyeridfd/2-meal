import streamlit as st
import pandas as pd
import numpy as np
import io
import openai # 실제 LLM 사용을 위한 라이브러리

# -------------------------------------------------------------------------
# 1. 데이터 로드 및 전처리
# -------------------------------------------------------------------------
@st.cache_data
def load_and_prep_data():
    try:
        menu_df = pd.read_csv('menu.csv')
        nutrient_df = pd.read_csv('nutrient.csv')
        category_df = pd.read_csv('category.csv')
        ingredient_df = pd.read_csv('ingredient.csv')
        
        # 고령자 데이터 헤더 자동 찾기
        patient_file = 'senior.csv'
        patient_df = pd.read_csv(patient_file, header=3)
        patient_df.columns = patient_df.columns.str.strip()

        menu_df.fillna(0, inplace=True)
        return menu_df, nutrient_df, category_df, ingredient_df, patient_df
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return None, None, None, None, None

def create_mock_inventory(ingredient_df):
    unique_ingredients = ingredient_df[['Ingredient', '품목코드', '단가(원/g)']].drop_duplicates()
    unique_ingredients['Stock_g'] = np.random.randint(0, 5000, size=len(unique_ingredients))
    return unique_ingredients

# -------------------------------------------------------------------------
# 2. [Real LLM] GPT API 호출 함수
# -------------------------------------------------------------------------
def get_gpt_response(api_key, system_role, user_prompt):
    """
    OpenAI GPT 모델을 실제로 호출하는 함수입니다.
    """
    if not api_key:
        return "⚠️ API Key가 입력되지 않았습니다. 사이드바에 키를 입력해주세요."
    
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o", # gpt-3.5-turbo 도 사용 가능
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"🚫 에러 발생: {str(e)}"

# -------------------------------------------------------------------------
# 3. 식단 최적화 엔진 (규칙 기반)
# -------------------------------------------------------------------------
def optimize_meal_plan(patient, master_menu, nutrient_df, category_df, ingredient_df):
    optimized_plan = []
    
    target_kcal = float(patient['체중']) * 30 / 3
    
    current_total_kcal = 0
    for m in master_menu:
        n = nutrient_df[nutrient_df['Menu'] == m]
        if not n.empty: current_total_kcal += n['에너지(kcal)'].values[0]
    
    ratio = target_kcal / current_total_kcal if current_total_kcal > 0 else 1.0
    ratio = max(0.7, min(ratio, 1.3))

    for menu in master_menu:
        cat_info = category_df[category_df['Menu'] == menu]
        cat = cat_info['Category'].values[0] if not cat_info.empty else "기타"
        
        nutri_info = nutrient_df[nutrient_df['Menu'] == menu]
        na_val = nutri_info['나트륨(mg)'].values[0] if not nutri_info.empty else 0
        kcal_val = nutri_info['에너지(kcal)'].values[0] if not nutri_info.empty else 0
        
        final_menu = menu
        final_amount_ratio = 1.0
        action_note = []

        # [규칙 A] 고혈압 대체
        if pd.notna(patient.get('고혈압')) and cat == '부찬' and na_val > 400:
            candidates = nutrient_df[nutrient_df['Menu'].isin(category_df[category_df['Category']=='부찬']['Menu'])]
            low_na_candidates = candidates[candidates['나트륨(mg)'] < 300]
            if not low_na_candidates.empty:
                final_menu = low_na_candidates.sample(1).iloc[0]['Menu']
                action_note.append("🔄 저염 대체")
                kcal_val = low_na_candidates[low_na_candidates['Menu']==final_menu]['에너지(kcal)'].values[0]

        # [규칙 B] 식감 변형
        texture_status = str(patient.get('현재식사현황', '일반'))
        if '죽' in texture_status and cat == '밥':
            final_menu = "흰죽"
            action_note.append("🥣 죽식 변경")
            final_amount_ratio = 1.5
        elif '다진' in texture_status and cat not in ['밥', '국', '죽']:
            action_note.append("🔪 다짐 조리")
        elif '갈' in texture_status and cat not in ['밥', '국', '죽']:
            action_note.append("🌪️ 갈기 조리")

        # [규칙 C] 양 조절
        if cat in ['밥', '국', '죽']:
            final_amount_ratio *= ratio
            if ratio != 1.0:
                action_note.append(f"⚖️ 양 {int(ratio*100)}%")

        ing_rows = ingredient_df[ingredient_df['Menu'] == final_menu]
        
        optimized_plan.append({
            'Category': cat,
            'Menu': final_menu,
            'Note': ", ".join(action_note),
            'Amount_Ratio': final_amount_ratio,
            'Ingredients': ing_rows,
            'Kcal': kcal_val * final_amount_ratio
        })

    return optimized_plan

# -------------------------------------------------------------------------
# 4. 발주 시스템
# -------------------------------------------------------------------------
def generate_order_sheet(all_patient_plans, current_inventory):
    total_requirements = {} 

    for plan in all_patient_plans:
        for menu_item in plan:
            ratio = menu_item['Amount_Ratio']
            ings = menu_item['Ingredients']
            for _, row in ings.iterrows():
                ing_name = row['Ingredient']
                required = row['Amount_g'] * ratio
                total_requirements[ing_name] = total_requirements.get(ing_name, 0) + required

    order_list = []
    inventory_dict = dict(zip(current_inventory['Ingredient'], current_inventory['Stock_g']))
    prices_dict = dict(zip(current_inventory['Ingredient'], current_inventory['단가(원/g)']))
    codes_dict = dict(zip(current_inventory['Ingredient'], current_inventory['품목코드']))

    for ing, needed_amount in total_requirements.items():
        stock = inventory_dict.get(ing, 0)
        needed_amount = np.ceil(needed_amount)
        if stock < needed_amount:
            to_order = needed_amount - stock
            price = prices_dict.get(ing, 0)
            order_list.append({
                '품목코드': codes_dict.get(ing, '-'),
                '품목명': ing,
                '현재고(g)': stock,
                '필요량(g)': needed_amount,
                '발주필요량(g)': to_order,
                '예상비용(원)': to_order * price
            })
            
    return pd.DataFrame(order_list)

# -------------------------------------------------------------------------
# 5. 메인 UI
# -------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="AI 요양원 통합 급식 관리")
    st.title("🏥 AI 요양원 통합 급식 시스템 (Real LLM Ver.)")
    
    menu_df, nutrient_df, category_df, ingredient_df, patient_df = load_and_prep_data()
    if menu_df is None: return
    
    if 'inventory' not in st.session_state:
        st.session_state['inventory'] = create_mock_inventory(ingredient_df)
    
    current_inventory = st.session_state['inventory']

    # --- 사이드바: 설정 및 API Key ---
    with st.sidebar:
        st.header("🔑 설정")
        # 실제 LLM 사용을 위해 키 입력받기
        api_key = st.text_input("OpenAI API Key", type="password", help="sk-로 시작하는 키를 입력하세요.")
        if not api_key:
            st.warning("키가 없으면 AI 기능이 작동하지 않습니다.")
        
        st.markdown("---")
        st.header("📅 날짜 선택")
        selected_date = st.selectbox("날짜", menu_df.columns[1:])
        
        st.markdown("---")
        st.subheader("📦 재고 현황")
        st.metric("총 품목", f"{len(current_inventory)} 개")

    tab1, tab2, tab3 = st.tabs(["👥 개인 맞춤 식단", "📦 재고 및 발주", "🤖 AI 영양사 상담"])

    # === TAB 1: 개인 맞춤 식단 ===
    with tab1:
        st.subheader(f"🍽️ {selected_date} 맞춤 식단표")
        master_menu = menu_df[selected_date].dropna().head(6).values
        selected_patient_name = st.selectbox("수급자 선택", patient_df['수급자명'].unique())
        patient_info = patient_df[patient_df['수급자명'] == selected_patient_name].iloc[0]

        optimized_plan = optimize_meal_plan(patient_info, master_menu, nutrient_df, category_df, ingredient_df)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(f"**{patient_info['수급자명']}**님 ({patient_info['현재식사현황']})")
            st.write(f"질환: 당뇨({patient_info.get('당뇨병','X')}), 고혈압({patient_info.get('고혈압','X')})")
            
        with col2:
            disp_data = []
            for item in optimized_plan:
                disp_data.append({
                    '구분': item['Category'],
                    '메뉴명': item['Menu'],
                    '변경사항': item['Note'],
                    '양(%)': f"{int(item['Amount_Ratio']*100)}%"
                })
            st.dataframe(pd.DataFrame(disp_data), use_container_width=True)

        st.markdown("---")
        # [Real LLM 기능] 선택한 메뉴에 대한 조리법 생성
        st.subheader("🍳 AI 조리 가이드 생성")
        target_menu_idx = st.selectbox("레시피를 생성할 메뉴를 선택하세요", range(len(optimized_plan)), format_func=lambda x: optimized_plan[x]['Menu'])
        
        if st.button("✨ 선택한 메뉴의 맞춤형 레시피 생성 (LLM 호출)"):
            target_item = optimized_plan[target_menu_idx]
            menu_name = target_item['Menu']
            notes = target_item['Note']
            
            # 프롬프트 구성
            system_role = "당신은 요양원 전문 조리장입니다. 고령자를 위한 안전하고 맛있는 조리법을 알려주세요."
            user_prompt = f"""
            메뉴명: {menu_name}
            대상 환자 특이사항: {notes} (예: 다짐식, 저염 등)
            환자 정보: {patient_info['현재식사현황']}, 연하장애 여부: {patient_info.get('연하장애','X')}
            
            위 조건을 만족하는 구체적인 조리 순서와 팁을 3단계로 요약해서 알려줘.
            특히 식감이나 염도 조절에 신경 써서 작성해줘.
            """
            
            with st.spinner("AI가 레시피를 작성 중입니다..."):
                recipe_result = get_gpt_response(api_key, system_role, user_prompt)
                st.success("작성 완료!")
                st.markdown(recipe_result)

    # === TAB 2: 재고 및 발주 ===
    with tab2:
        st.subheader("🛒 자동 발주 시스템")
        if st.button("🚀 전체 환자 발주서 생성"):
            all_plans = []
            bar = st.progress(0)
            for i, (_, p_info) in enumerate(patient_df.iterrows()):
                all_plans.append(optimize_meal_plan(p_info, master_menu, nutrient_df, category_df, ingredient_df))
                bar.progress((i+1)/len(patient_df))
            
            order_df = generate_order_sheet(all_plans, current_inventory)
            st.dataframe(order_df)
            
            # 엑셀 다운로드
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                order_df.to_excel(writer, index=False)
            st.download_button("📥 발주서 엑셀 다운로드", buffer.getvalue(), f"발주서_{selected_date}.xlsx")

    # === TAB 3: AI 영양사 상담 (Real LLM) ===
    with tab3:
        st.subheader("🤖 AI 영양사 상담소")
        st.markdown("현재 **재고 현황**을 기반으로 메뉴 추천이나 영양 상담을 받을 수 있습니다.")
        
        user_query = st.text_input("질문을 입력하세요", placeholder="예: 재고 중에 감자가 많은데 고혈압 환자용 간식 추천해줘")
        
        if st.button("질문하기"):
            # 현재 재고 정보 중 많은 것 상위 5개를 추출해서 프롬프트에 제공
            top_stocks = current_inventory.sort_values('Stock_g', ascending=False).head(5)['Ingredient'].tolist()
            
            system_role = "당신은 데이터 기반의 요양원 영양사입니다. 보유 재고를 고려하여 실질적인 조언을 해주세요."
            user_prompt = f"""
            [현재 보유 재고 상위 품목]
            {', '.join(top_stocks)}
            
            [질문]
            {user_query}
            
            답변은 친절하게 하고, 가능한 재고를 활용하는 방향으로 제안해줘.
            """
            
            with st.spinner("AI가 고민 중입니다..."):
                answer = get_gpt_response(api_key, system_role, user_prompt)
                st.chat_message("assistant").write(answer)

if __name__ == "__main__":
    main()
