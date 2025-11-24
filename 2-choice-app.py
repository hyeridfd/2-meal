import streamlit as st
import pandas as pd
import numpy as np
import io

# -------------------------------------------------------------------------
# 1. 데이터 로드 및 가상 재고 생성
# -------------------------------------------------------------------------
@st.cache_data
def load_and_prep_data():
    try:
        # 데이터 로드 (파일명은 사용자의 환경에 맞게 수정 필요)
        menu_df = pd.read_csv('menu.csv')
        nutrient_df = pd.read_csv('nutrient.csv')
        category_df = pd.read_csv('category.csv')
        ingredient_df = pd.read_csv('ingredient.csv')
        
        # 고령자 데이터 로드 (헤더 자동 찾기)
        patient_file = ‘senior.csv'
        patient_df = pd.read_csv(patient_file, header=3)
        patient_df.columns = patient_df.columns.str.strip()

        menu_df.fillna(0, inplace=True)
        return menu_df, nutrient_df, category_df, ingredient_df, patient_df
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return None, None, None, None, None

def create_mock_inventory(ingredient_df):
    """
    현재 보유 재고 데이터가 없으므로, ingredient DB를 기반으로 가상의 재고를 생성합니다.
    실제 서비스에서는 이 부분을 '재고 관리 엑셀 파일'을 읽어오는 코드로 대체하면 됩니다.
    """
    unique_ingredients = ingredient_df[['Ingredient', '품목코드', '단가(원/g)']].drop_duplicates()
    # 랜덤하게 재고량 부여 (0 ~ 5000g)
    unique_ingredients['Stock_g'] = np.random.randint(0, 5000, size=len(unique_ingredients))
    return unique_ingredients

# -------------------------------------------------------------------------
# 2. [핵심 로직] 개인 맞춤형 식단 설계 엔진
# -------------------------------------------------------------------------
def optimize_meal_plan(patient, master_menu, nutrient_df, category_df, ingredient_df):
    """
    환자 1명의 한 끼 식단을 설계합니다. (양 조절, 메뉴 교체, 식감 변형)
    """
    optimized_plan = []
    
    # 1. 환자 필요 칼로리 계산 (간이 공식: 체중 * 30) - 실제로는 Harris-Benedict 등 사용
    target_kcal = float(patient['체중']) * 30 / 3 # 한 끼 목표
    
    # 현재 마스터 메뉴의 총 칼로리 계산
    current_total_kcal = 0
    for m in master_menu:
        n = nutrient_df[nutrient_df['Menu'] == m]
        if not n.empty: current_total_kcal += n['에너지(kcal)'].values[0]
    
    # 칼로리 조정 비율 (단, 0.7 ~ 1.3 범위로 제한하여 너무 적거나 많지 않게)
    ratio = target_kcal / current_total_kcal if current_total_kcal > 0 else 1.0
    ratio = max(0.7, min(ratio, 1.3))

    for menu in master_menu:
        # 정보 조회
        cat_info = category_df[category_df['Menu'] == menu]
        cat = cat_info['Category'].values[0] if not cat_info.empty else "기타"
        
        nutri_info = nutrient_df[nutrient_df['Menu'] == menu]
        na_val = nutri_info['나트륨(mg)'].values[0] if not nutri_info.empty else 0
        kcal_val = nutri_info['에너지(kcal)'].values[0] if not nutri_info.empty else 0
        
        final_menu = menu
        final_amount_ratio = 1.0 # 기본 양 (100%)
        action_note = []

        # --- [A] 질환 기반 메뉴 교체 (고혈압 -> 저염 부찬) ---
        if pd.notna(patient.get('고혈압')) and cat == '부찬' and na_val > 400:
            # 같은 카테고리 내 저염 메뉴 검색
            candidates = nutrient_df[nutrient_df['Menu'].isin(category_df[category_df['Category']=='부찬']['Menu'])]
            low_na_candidates = candidates[candidates['나트륨(mg)'] < 300]
            
            if not low_na_candidates.empty:
                final_menu = low_na_candidates.sample(1).iloc[0]['Menu']
                action_note.append("🔄 저염 대체")
                # 교체된 메뉴의 영양정보로 업데이트
                kcal_val = low_na_candidates[low_na_candidates['Menu']==final_menu]['에너지(kcal)'].values[0]

        # --- [B] 저작 단계별 식감 변형 ---
        texture_status = str(patient.get('현재식사현황', '일반'))
        
        if '죽' in texture_status and cat == '밥':
            final_menu = "흰죽"
            action_note.append("🥣 죽식 변경")
            final_amount_ratio = 1.5 # 죽은 밥보다 부피가 크므로 양 조정
            
        elif '다진' in texture_status and cat not in ['밥', '국', '죽']:
            action_note.append("🔪 다짐 조리")
            
        elif '갈' in texture_status and cat not in ['밥', '국', '죽']: # 갈찬
            action_note.append("🌪️ 갈기 조리")

        # --- [C] 칼로리 기반 양 조절 (밥, 국 위주로 조절) ---
        # 반찬은 조리 공정상 개별 양 조절이 어려우므로 밥/국으로 칼로리 밸런스 맞춤
        if cat in ['밥', '국', '죽']:
            final_amount_ratio *= ratio
            if ratio != 1.0:
                action_note.append(f"⚖️ 양 {int(ratio*100)}%")

        # 결과 저장
        # 해당 메뉴에 필요한 재료 목록 가져오기
        ing_rows = ingredient_df[ingredient_df['Menu'] == final_menu]
        
        optimized_plan.append({
            'Category': cat,
            'Menu': final_menu,
            'Note': ", ".join(action_note),
            'Amount_Ratio': final_amount_ratio,
            'Ingredients': ing_rows, # 재료 데이터프레임 통째로 저장
            'Kcal': kcal_val * final_amount_ratio
        })

    return optimized_plan

# -------------------------------------------------------------------------
# 3. 발주 시스템 (소요량 계산 -> 재고 차감 -> 발주서 생성)
# -------------------------------------------------------------------------
def generate_order_sheet(all_patient_plans, current_inventory):
    """
    모든 환자의 식단계획을 합쳐서 총 식자재 소요량을 계산하고 발주서를 만듭니다.
    """
    total_requirements = {} # {재료명: 필요량_g}

    # 1. 소요량 집계
    for plan in all_patient_plans: # 환자별
        for menu_item in plan: # 메뉴별
            ratio = menu_item['Amount_Ratio']
            ings = menu_item['Ingredients']
            
            for _, row in ings.iterrows():
                ing_name = row['Ingredient']
                base_amount = row['Amount_g']
                required = base_amount * ratio
                
                if ing_name in total_requirements:
                    total_requirements[ing_name] += required
                else:
                    total_requirements[ing_name] = required

    # 2. 재고 비교 및 발주 리스트 생성
    order_list = []
    
    inventory_dict = dict(zip(current_inventory['Ingredient'], current_inventory['Stock_g']))
    prices_dict = dict(zip(current_inventory['Ingredient'], current_inventory['단가(원/g)']))
    codes_dict = dict(zip(current_inventory['Ingredient'], current_inventory['품목코드']))

    for ing, needed_amount in total_requirements.items():
        stock = inventory_dict.get(ing, 0) # 재고 없으면 0
        needed_amount = np.ceil(needed_amount) # 소수점 올림
        
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
# 4. 메인 UI
# -------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="AI 요양원 통합 급식 관리")
    st.title("🏥 AI 요양원 통합 급식 시스템 (식단+재고+발주)")
    
    # 데이터 로드
    menu_df, nutrient_df, category_df, ingredient_df, patient_df = load_and_prep_data()
    if menu_df is None: return
    
    # 가상 재고 생성 (세션 상태에 저장하여 유지)
    if 'inventory' not in st.session_state:
        st.session_state['inventory'] = create_mock_inventory(ingredient_df)
    
    current_inventory = st.session_state['inventory']

    # --- 사이드바 ---
    with st.sidebar:
        st.header("📅 식단 및 발주 설정")
        selected_date = st.selectbox("날짜 선택", menu_df.columns[1:])
        st.markdown("---")
        st.subheader("📦 재고 현황 요약")
        st.metric("총 등록 품목", f"{len(current_inventory)} 개")
        low_stock = len(current_inventory[current_inventory['Stock_g'] < 1000])
        st.metric("부족 품목 (1kg 미만)", f"{low_stock} 개", delta_color="inverse")

    # --- 탭 구성 ---
    tab1, tab2, tab3 = st.tabs(["👥 개인 맞춤 식단", "📦 재고 및 발주 관리", "🤖 AI 조리 비서"])

    # === TAB 1: 개인 맞춤 식단 ===
    with tab1:
        st.subheader(f"🍽️ {selected_date} 환자별 맞춤 식단표")
        
        # 날짜별 마스터 메뉴
        master_menu = menu_df[selected_date].dropna().head(6).values
        
        # 환자 선택
        selected_patient_name = st.selectbox("수급자 상세 조회", patient_df['수급자명'].unique())
        patient_info = patient_df[patient_df['수급자명'] == selected_patient_name].iloc[0]

        # 식단 최적화 실행
        optimized_plan = optimize_meal_plan(patient_info, master_menu, nutrient_df, category_df, ingredient_df)
        
        # 결과 표시
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.info(f"**{patient_info['수급자명']}**님 정보")
            st.write(f"- 질환: 당뇨({patient_info.get('당뇨병','X')}), 고혈압({patient_info.get('고혈압','X')})")
            st.write(f"- 식사형태: {patient_info['현재식사현황']}")
            st.write(f"- 목표 칼로리: 약 {int(float(patient_info['체중'])*10)}kcal")
            
        with col2:
            st.markdown("#### ✅ 맞춤 식단 및 조리 지시")
            
            # DataFrame으로 변환해서 이쁘게 보여주기
            disp_data = []
            for item in optimized_plan:
                disp_data.append({
                    '구분': item['Category'],
                    '메뉴명': item['Menu'],
                    '조리/배식 지침': item['Note'],
                    '제공량(비율)': f"{int(item['Amount_Ratio']*100)}%"
                })
            st.dataframe(pd.DataFrame(disp_data), use_container_width=True)

    # === TAB 2: 재고 및 발주 관리 ===
    with tab2:
        st.subheader("🛒 자동 발주 시스템")
        st.write("선택한 날짜의 **모든 환자 식단**을 분석하여 부족한 식자재를 계산합니다.")
        
        if st.button("🚀 전체 환자 발주서 생성하기"):
            # 1. 모든 환자에 대해 식단 최적화 수행
            all_plans = []
            progress_bar = st.progress(0)
            
            total_patients = len(patient_df)
            for i, (_, p_info) in enumerate(patient_df.iterrows()):
                plan = optimize_meal_plan(p_info, master_menu, nutrient_df, category_df, ingredient_df)
                all_plans.append(plan)
                progress_bar.progress((i + 1) / total_patients)
            
            # 2. 발주서 생성
            order_df = generate_order_sheet(all_plans, current_inventory)
            
            st.success("발주서 생성이 완료되었습니다!")
            
            if not order_df.empty:
                st.dataframe(order_df)
                st.metric("총 예상 발주 금액", f"{int(order_df['예상비용(원)'].sum()):,} 원")
                
                # 엑셀 다운로드 버튼
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    order_df.to_excel(writer, sheet_name='발주서', index=False)
                
                st.download_button(
                    label="📥 발주서 엑셀 다운로드",
                    data=buffer.getvalue(),
                    file_name=f"발주서_{selected_date}.xlsx",
                    mime="application/vnd.ms-excel"
                )
            else:
                st.info("현재 재고가 충분하여 발주할 품목이 없습니다.")

    # === TAB 3: LLM AI 활용 ===
    with tab3:
        st.subheader("🤖 LLM 식단 매니저")
        st.info("이곳은 생성형 AI(LLM)가 식단 구성이나 대체 메뉴에 대해 조언해주는 공간입니다.")
        
        user_query = st.text_input("질문 예시: 고혈압 환자인데 멸치볶음 대신 뭘 주면 좋을까? 우리 재고 중에 추천해줘.")
        if st.button("AI에게 물어보기"):
            # 실제 연결 시 여기에 OpenAI API 호출 코드 삽입
            # prompt = f"현재 재고: {current_inventory.sample(10)['Ingredient'].tolist()}... 질문: {user_query}"
            st.markdown(f"""
            **🤖 AI 답변 (시뮬레이션):**
            
            고혈압 환자에게 멸치볶음은 나트륨 함량이 높아 부담될 수 있습니다. 
            현재 보유하신 재고 중 **'두부'**와 **'양파'**가 넉넉하네요.
            
            추천 대체 메뉴: **두부 양파 조림 (저염 간장 사용)**
            1. 두부를 깍둑썰기하여 물기를 제거합니다.
            2. 양파와 함께 들기름에 살짝 볶아 풍미를 높입니다.
            3. 일반 간장 대신 저염 간장을 소량만 사용하여 간을 맞춥니다.
            
            이렇게 하면 나트륨은 낮추고 단백질 섭취는 유지할 수 있습니다.
            """)

if __name__ == "__main__":
    main()
