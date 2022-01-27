<업무 분담3>
각자 모델링한 파트 * 4개의 열(Low , High, Adj Close, Change)
ex) 종가의(Adj Close) 경우


각 last_30 데이터프레임에 추가시키고(어제 종가까지의 데이터가 있음)
다시 마지막 30개의 데이터를 예측시켜서 표에다가 입력
다음날 XX:XX시에 실제 값과 비교하여
24개까지 예측하고 난 후에 정답을 매김
각 모델의 정답률을 보여주고
정답률이 80프로 이상인 모델은(종목-열) 따로 '유망종목' 등등의 이름으로 보여주기.

'실행' 하면 기존의 Df마지막 날 이후부터(end_day + today) 어제까지의 종가데이터를 하져옴
 * 문제점: Yahoo Finance의 데이터 업데이터 시간이 정확한가..

ex) High의 경우
ex) Low 의 경우
ex) Change 의 경우

* low high close는 정답지표를 어떤 기준으로 할지... 흠










# < 업무분담 - 2 >
# 예측 할 컬럼 4개 : Low , High, Adj Close, Change
# 1. 마지막 벡테스팅용 30개 데이터는 따로 만들기 (예시 : last_30_currency )
# 1. minmaxscaler 만들기( 이름 예시 : currency_minmaxscaler)
# 2. 30개씩 예측시키기 - train_test_split까지(test_size = 0.2) for문돌릴 때, 전체행 갯수 -30
# 3. 모델링은 갑래님 모델과 같은 구조로
# 4. minmaxscaler, 모델.h5, train_test_split, last_30개 데이터를 만들 때 저장이름
#    : 자기파트 섹터_model.h5, 자기파트 섹터_last_30 등  (예시 : futures_minmaxscaler)
# 코랩과 파이참을 이용해서 ....!



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