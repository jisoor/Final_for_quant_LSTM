< 업무분담 - 2 >
예측 할 컬럼 4개 : Low , High, Adj Close, Change
1. 마지막 벡테스팅용 30개 데이터는 따로 만들기 (예시 : last_30_currency )
1. minmaxscaler 만들기( 이름 예시 : currency_minmaxscaler)
2. 30개씩 예측시키기 - train_test_split까지(test_size = 0.2) for문돌릴 때, 전체행 갯수 -30
3. 모델링은 갑래님 모델과 같은 구조로
4. minmaxscaler, 모델.h5, train_test_split, last_30개 데이터를 만들 때 저장이름
   : 자기파트 섹터_model.h5, 자기파트 섹터_last_30 등  (예시 : futures_minmaxscaler)
코랩과 파이참을 이용해서 ....!



# < 업무분담 - 1>
# 1. 맡는 파트 -> 갑래: futures, 영준: currencies, 지수:world indices 안의 모든 종목들을
# 2. 다운로드 받아서 (yf.download('티커명', '시작날짜', '마지막 날짜'
# 3. Date를 index 로 가지고  ' low', 'High', 'Adj Close' 의 3개 열로로 구성된 데이터프레임 만들기 (for 문 사용)
# 4. 하나의 열을 더 추가할 건데 name = Change, 이고 Adj Close열에다가 pct_change를 쓴다.(전날에 비해 등락율)
# 5. 2007-01-01 ~ 2022-01-25 인지 확인.
# 6. Nan 값 처리는 fillna로 이전값을 취함.
# 7. Nan값 분포에 대한 plot도 작성
# 8. 저장은 world_indices_df.csv 등 의 형식으로
# 9. 완성 되면 코드랑 csv파일을 드라이브에 올려줍니다(코드는 저한테 fork하고 request요청해서 직접 이 깃허브에 추가해도되요)
#
# 시간이 촉박해서 최대한 내일 오후 이전까지 완성해주세요~!