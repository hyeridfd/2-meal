import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import random

# -------------------------------------------------------------------------
# 1. 데이터 로드 (오류 수정 버전)
# -------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        menu_df = pd.read_csv('menu.csv')
        nutrient_df = pd.read_csv('nutrient.csv')
        category_df = pd.read_csv('category.csv')
        
        # 고령자 데이터 헤더 자동 찾기 로직
        patient_file_name = 'senior.csv'
        patient_df = pd.read_csv(patient_file_name, header=3)            
        menu_df.fillna(0, inplace=True)
        return menu_df, nutrient_df, category_df, patient_df
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return None, None, None, None

# -------------------------------------------------------------------------
# 2. [핵심] 메뉴 대체 알고리즘
# -------------------------------------------------------------------------
def find_substitute_menu(current_menu, category, condition, nutrient_df, category_df):
    """
    현재 메뉴가 환자 조건에 맞지 않으면, 같은 카테고리의 '건강한 메뉴'를 DB에서 찾아서 바꿔줍니다.
    """
    # 1. 같은 카테고리(예: 국, 주찬)의 모든 메뉴 리스트 확보
    same_category_menus = category_df[category_df['Category'] == category]['Menu'].unique()
    
    # 2. 해당 메뉴들의 영양 정보 가져오기
    candidates = nutrient_df[nutrient_df['Menu'].isin(same_category_menus)].copy()
    
    if candidates.empty:
        return current_menu, "대체 메뉴 없음"

    # 3. 질환별 필터링 (여기가 '지능'이 들어가는 부분!)
    recommended = pd.DataFrame()
    reason = ""

    if condition == '고혈압':
        # 나트륨 400mg 미만인 메뉴 찾기
        recommended = candidates[candidates['나트륨(mg)'] < 400]
        reason = "저염식 대체"
    elif condition == '당뇨':
        # 탄수화물 40g 미만인 메뉴 찾기 (반찬 기준)
        recommended = candidates[candidates['탄수화물(g)'] < 40]
        reason = "저탄수 대체"
    
    # 4. 대체 메뉴 선정
    if not recommended.empty:
        # 조건에 맞는 메뉴 중 하나를 랜덤으로 추천 (매번 다르게)
        new_menu = recommended.sample(1).iloc[0]['Menu']
        # 원래 메뉴와 다를 때만 반환
        if new_menu != current_menu:
            return new_menu, f"{reason} (Na: {recommended[recommended['Menu']==new_menu]['나트륨(mg)'].values[0]}mg)"
    
    return current_menu, "" # 대체할 게 없으면 원래 메뉴 유지

# -------------------------------------------------------------------------
# 3. 식단 변환 로직 (메뉴 대체 기능 추가)
# -------------------------------------------------------------------------
def personalize_menu_advanced(master_menu_list, patient_info, nutrient_df, category_df):
    final_menu_list = []
    
    # 환자 상태 파악
    is_hypertension = pd.notna(patient_info.get('고혈압', None))
    is_diabetes = pd.notna(patient_info.get('당뇨병', None))
    
    for menu in master_menu_list:
        # 현재 메뉴 정보 조회
        cat_row = category_df[category_df['Menu'] == menu]
        cat = cat_row['Category'].values[0] if not cat_row.empty else "기타"
        
        nutri_row = nutrient_df[nutrient_df['Menu'] == menu]
        current_na = nutri_row['나트륨(mg)'].values[0] if not nutri_row.empty else 0
        
        final_menu = menu
        note = ""
        is_changed = False

        # --- [로직 1] 고혈압 환자인데 나트륨이 600mg 넘는 메뉴가 있다? -> 교체! ---
        if is_hypertension and current_na > 600:
            final_menu, change_reason = find_substitute_menu(menu, cat, '고혈압', nutrient_df, category_df)
            if final_menu != menu:
                note = f"🔄 {change_reason}"
                is_changed = True
        
        # --- [로직 2] 당뇨 환자인데 주찬이 너무 달다? (예시 로직) -> 교체! ---
        # (여기서는 예시로 로직 1과 비슷하게 구현 가능)
        
        # --- [로직 3] 연하장애 (이건 메뉴 교체보다는 조리법 변경이 맞음) ---
        if pd.notna(patient_info.get('연하장애', None)):
            if cat in ['밥']:
                final_menu = "흰죽"
                note += " (점도 조절)"
                is_changed = True
            elif cat not in ['국', '죽']:
                note += " (다짐/갈기 조리)"
                is_changed = True

        final_menu_list.append({
            'Category': cat,
            'Original': menu,
            'Final': final_menu,
            'Note': note,
            'Changed': is_changed
        })
        
    return pd.DataFrame(final_menu_list)

# -------------------------------------------------------------------------
# 4. 메인 UI
# -------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="AI 급식 메뉴 대체 시스템")
    st.title("🥗 질환 맞춤형 메뉴 자동 대체(Substitution) 시스템")
    st.markdown("---")

    menu_df, nutrient_df, category_df, patient_df = load_data()
    if menu_df is None: return

    with st.sidebar:
        selected_date = st.selectbox("날짜 선택", menu_df.columns[1:])
        selected_patient = st.selectbox("수급자 선택", patient_df['수급자명'].unique())
    
    patient_info = patient_df[patient_df['수급자명'] == selected_patient].iloc[0]
    
    # 환자 정보 표시
    st.info(f"**{patient_info['수급자명']}**님 (고혈압: {'O' if pd.notna(patient_info.get('고혈압')) else 'X'}, 당뇨: {'O' if pd.notna(patient_info.get('당뇨병')) else 'X'})")

    # 데이터 처리
    master_menu = menu_df[selected_date].dropna().head(6).values
    result_df = personalize_menu_advanced(master_menu, patient_info, nutrient_df, category_df)

    # 결과 시각화
    st.subheader(f"🔄 {selected_date} 식단 변환 결과")
    
    # 컬럼 스타일링을 위한 함수
    def highlight_change(row):
        return ['background-color: #d1e7dd' if row['Changed'] else '' for _ in row]

    st.dataframe(
        result_df[['Category', 'Original', 'Final', 'Note']],
        use_container_width=True,
        height=400
    )

    # 전후 비교 요약
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ❌ 변경 전 (Original)")
        for m in master_menu:
            st.text(f"- {m}")
    with col2:
        st.markdown("### ✅ 변경 후 (Personalized)")
        for idx, row in result_df.iterrows():
            if row['Changed']:
                st.markdown(f"- **{row['Final']}** :red[[변경됨]]")
            else:
                st.text(f"- {row['Final']}")

if __name__ == "__main__":
    main()
