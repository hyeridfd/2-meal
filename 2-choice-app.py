import streamlit as st
import pandas as pd
import random
import datetime

# -------------------------------------------------------------------------
# 1. 데이터 로드 및 전처리
# -------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        menu_df = pd.read_csv('menu.csv')
        category_df = pd.read_csv('category.csv')
        
        # 고령자 데이터 헤더 자동 찾기
        patient_file = 'senior.csv'
        patient_df = pd.read_csv(patient_file, header=3)
        patient_df.columns = patient_df.columns.str.strip()

        menu_df.fillna(0, inplace=True)
        return menu_df, category_df, patient_df
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return None, None, None

# -------------------------------------------------------------------------
# 2. 부찬 속성 분류기 (육고기/해산물/채소)
# -------------------------------------------------------------------------
def classify_side_dishes(category_df):
    """
    모든 '부찬'을 재료 특성에 따라 분류합니다.
    실제 서비스에서는 이 부분을 LLM에게 맡기면 훨씬 정확합니다.
    여기서는 빠른 데모를 위해 키워드 기반으로 분류합니다.
    """
    side_dishes = category_df[category_df['Category'] == '부찬']['Menu'].unique()
    
    classified_db = {
        '육고기': [],
        '해산물': [],
        '채소': []
    }
    
    # 키워드 사전
    meat_keywords = ['소고기', '돈육', '돼지', '햄', '베이컨', '소세지', '장조림', '닭', '미트볼', '계란', '메추리알']
    sea_keywords = ['멸치', '어묵', '김', '미역', '새우', '오징어', '참치', '명태', '코다리', '굴']
    # 나머지는 채소로 간주
    
    for menu in side_dishes:
        is_classified = False
        
        # 육류 체크
        for k in meat_keywords:
            if k in menu:
                classified_db['육고기'].append(menu)
                is_classified = True
                break
        
        # 해산물 체크 (육류가 아니면)
        if not is_classified:
            for k in sea_keywords:
                if k in menu:
                    classified_db['해산물'].append(menu)
                    is_classified = True
                    break
        
        # 둘 다 아니면 채소
        if not is_classified:
            classified_db['채소'].append(menu)
            
    return classified_db

# -------------------------------------------------------------------------
# 3. 한 달치 식단 생성 엔진
# -------------------------------------------------------------------------
def generate_monthly_plan(master_menu_df, category_df, side_dish_db, preference):
    """
    1주일치 데이터를 4번 반복하여 4주(28일) 식단을 생성하되,
    부찬만 선호도에 맞춰 교체합니다.
    """
    
    # 1주일치 날짜 컬럼들
    base_dates = master_menu_df.columns[1:] 
    
    monthly_plan = []
    
    # 4주 반복 (Week 1 ~ Week 4)
    for week in range(4): 
        for date_col in base_dates:
            # 날짜 계산 (가상)
            base_dt = datetime.datetime.strptime(date_col.split(' ')[0], "%Y-%m-%d")
            new_date = base_dt + datetime.timedelta(weeks=week)
            date_str = new_date.strftime("%Y-%m-%d (%a)")
            
            # 해당 날짜의 마스터 메뉴 가져오기 (결측치 제거)
            daily_menus = master_menu_df[date_col].dropna().tolist()
            
            # 하루 식단 구성 (아침, 점심, 저녁 중 '조식' 6개만 예시로 사용)
            # 실제 데이터에 따라 슬라이싱 조정 필요
            daily_menus = daily_menus[:6] 
            
            day_plan = {
                '날짜': date_str,
                '밥': '', '국': '', '주찬': '', '김치': '', 
                '부찬': [], '원래부찬': []
            }
            
            for menu in daily_menus:
                # 카테고리 확인
                cat_row = category_df[category_df['Menu'] == menu]
                if cat_row.empty: continue
                cat = cat_row['Category'].values[0]
                
                if cat == '부찬':
                    day_plan['원래부찬'].append(menu)
                    
                    # [핵심 로직] 선호도 반영 교체
                    # 현재 부찬이 선호도 그룹에 속해있으면 유지, 아니면 교체
                    if menu in side_dishes_by_type[preference]:
                        day_plan['부찬'].append(menu) # 운 좋게 취향 일치 -> 유지
                    else:
                        # 취향에 맞는 다른 반찬 랜덤 추출 (재고/계절 고려 가능)
                        substitute = random.choice(side_dishes_by_type[preference])
                        day_plan['부찬'].append(f"{substitute} (🔄교체)")
                        
                elif cat in day_plan:
                    day_plan[cat] = menu
            
            # 부찬 리스트를 문자열로 변환
            day_plan['부찬'] = ", ".join(day_plan['부찬'])
            day_plan['원래부찬'] = ", ".join(day_plan['원래부찬'])
            
            monthly_plan.append(day_plan)
            
    return pd.DataFrame(monthly_plan)

# -------------------------------------------------------------------------
# 4. 메인 UI
# -------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="모듈형 맞춤 식단 생성기")
st.title("🗓️ 1개월치 모듈형 맞춤 식단 생성기")
st.markdown("""
- **공통:** 밥, 국, 주찬, 김치 (모두 동일)
- **개인화:** **부찬(Side Dish)**만 선호도(육고기/해산물/채소)에 따라 자동 변경
""")

# 데이터 로드
menu_df, category_df, patient_df = load_data()

if menu_df is not None:
    # 1. 부찬 DB 분류 실행
    side_dishes_by_type = classify_side_dishes(category_df)

    with st.sidebar:
        st.header("👤 대상자 설정")
        selected_patient = st.selectbox("수급자 선택", patient_df['수급자명'].unique())
        
        st.markdown("---")
        st.header("❤️ 선호도 조사")
        st.write("부찬(밑반찬)으로 어떤 종류를 선호하시나요?")
        preference = st.radio(
            "선호 식재료 선택",
            ('육고기', '채소', '해산물'),
            index=1
        )
        
        st.info(f"선택하신 **[{preference}]** 위주로 한 달 식단을 구성합니다.")
        
        # 디버깅용: 분류된 메뉴 보여주기
        with st.expander("분류된 부찬 DB 확인"):
            st.write(side_dishes_by_type[preference])

    # 2. 식단 생성
    final_plan_df = generate_monthly_plan(menu_df, category_df, side_dishes_by_type, preference)

    # 3. 결과 시각화
    st.subheader(f"📅 {selected_patient}님을 위한 4주 맞춤 식단표")
    
    # 데이터프레임 스타일링 (변경된 부찬 강조)
    def highlight_change(val):
        color = '#e6fffa' if '🔄' in str(val) else ''
        return f'background-color: {color}'

    st.dataframe(
        final_plan_df[['날짜', '밥', '국', '주찬', '김치', '부찬']],
        use_container_width=True,
        height=600,
        column_config={
            "날짜": st.column_config.TextColumn("날짜", width="medium"),
            "부찬": st.column_config.TextColumn("부찬 (맞춤형)", width="large"),
        }
    )
    
    # 4. 통계 및 요약
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        total_days = len(final_plan_df)
        changed_cnt = final_plan_df['부찬'].str.contains('🔄').sum()
        st.metric("식단 생성 기간", f"{total_days}일 (4주)")
        
    with col2:
        st.metric("취향 반영 교체 횟수", f"{changed_cnt}회 / {total_days}끼")
        st.caption("※ 원래 식단이 이미 취향과 맞으면 교체하지 않습니다.")

    # 5. 엑셀 다운로드
    import io
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        final_plan_df.to_excel(writer, index=False)
        
    st.download_button(
        label="📥 1개월 식단표 엑셀 다운로드",
        data=buffer.getvalue(),
        file_name=f"{selected_patient}_1개월_맞춤식단({preference}).xlsx",
        mime="application/vnd.ms-excel"
    )
