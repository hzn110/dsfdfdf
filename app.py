import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MOVIES_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"
DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"

st.set_page_config(page_title="영화 흥행 예측기", page_icon="🎬", layout="wide")

st.title("🎬 영화 흥행 예측기")
st.caption("KOBIS 영화별 정보로 총 관객 수를 다중 회귀로 예측합니다.")

@st.cache_data
def load_data():
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8")
    daily = pd.read_csv(DAILY_URL, encoding="utf-8")

    # 혹시 모를 앞뒤 공백/BOM 제거
    movies.columns = movies.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    daily.columns = daily.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    return movies, daily

try:
    movies, daily = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

required_movies = {"movieCd", "total_audi"}
required_daily = {"날짜"}
if not required_movies.issubset(movies.columns):
    st.error(f"영화별 표에 필요한 열이 없습니다: {sorted(required_movies - set(movies.columns))}")
    st.stop()
if not required_daily.issubset(daily.columns):
    st.error(f"일별 표에 필요한 열이 없습니다: {sorted(required_daily - set(daily.columns))}")
    st.stop()

# 영화코드 기준으로 정렬하고, 영화별 표의 모든 영화를 사용
movies = movies.copy()
movies["movieCd"] = movies["movieCd"].astype(str).str.strip()
movies = movies.sort_values("movieCd", kind="stable").reset_index(drop=True)

# 일별 데이터에서 기준 기간 계산
date_series = pd.to_datetime(
    daily["날짜"].astype(str).str.replace(r"\.0$", "", regex=True),
    format="%Y%m%d",
    errors="coerce",
)
valid_dates = date_series.dropna()

if len(valid_dates):
    period_text = f"{valid_dates.min():%Y-%m-%d} ~ {valid_dates.max():%Y-%m-%d}"
else:
    period_text = "기간을 확인할 수 없음"

# 화면에 영화별 표의 맨 위 행을 그대로 표시
st.subheader("영화별 표")
st.dataframe(movies, use_container_width=True, hide_index=True)

if len(movies) == 0:
    st.warning("영화별 데이터가 없습니다.")
    st.stop()

st.subheader("영화별 표의 첫 번째 행")
st.dataframe(movies.head(1), use_container_width=True, hide_index=True)

# 사용 가능한 설명변수
# movieCd는 식별자이고 total_audi는 정답이므로 기본적으로 제외
candidate_features = [c for c in movies.columns if c not in {"movieCd", "total_audi"}]

st.sidebar.header("모델 설정")
default_features = [
    c for c in [
        "openDt", "genre", "nation", "first_scrn", "first_show",
        "first_date", "peak", "first_week_audi", "days_in_top10"
    ] if c in candidate_features
]

selected_features = []
for feature in candidate_features:
    checked = st.sidebar.checkbox(
        feature,
        value=(feature in default_features),
        key=f"feature_{feature}",
    )
    if checked:
        selected_features.append(feature)

st.sidebar.markdown("---")
st.sidebar.write("**고정 평가 방식**")
st.sidebar.write("영화코드 순으로 정렬한 뒤, 10편마다 앞의 3편을 테스트용으로 사용합니다.")
st.sidebar.write("나머지 영화로 학습하며, 모든 영화 행을 사용합니다.")

if not selected_features:
    st.warning("왼쪽에서 학습 변수를 하나 이상 선택하세요.")
    st.stop()

# 숫자처럼 들어온 값은 숫자로 변환할 수 있으면 숫자로 취급.
# 날짜 열은 숫자형 '날짜의 일수'로 변환해 회귀에 사용.
X = movies[selected_features].copy()
y = pd.to_numeric(movies["total_audi"], errors="coerce")

for col in X.columns:
    if col in {"openDt", "first_date"}:
        parsed = pd.to_datetime(
            X[col].astype(str).str.replace(r"\.0$", "", regex=True),
            format="%Y%m%d",
            errors="coerce",
        )
        # 전체 데이터에서 가장 이른 날짜를 0일로 두는 상대 날짜
        base = parsed.min()
        if pd.notna(base):
            X[col] = (parsed - base).dt.days.astype(float)
        else:
            X[col] = np.nan
    elif X[col].dtype == "object":
        # 실제 숫자열이 문자열로 저장되어 있으면 숫자로 변환
        numeric = pd.to_numeric(X[col], errors="coerce")
        if numeric.notna().mean() >= 0.95:
            X[col] = numeric

# 정답이 없는 행은 모델 학습/평가에서 제외하지만,
# 영화별 표 자체는 위에서 원본 그대로 전부 표시함.
valid_target = y.notna()
X = X.loc[valid_target].reset_index(drop=True)
y = y.loc[valid_target].reset_index(drop=True)
model_movies = movies.loc[valid_target].reset_index(drop=True)

n = len(model_movies)
positions = np.arange(n)

# 10편 단위로 앞 3편 테스트: 0,1,2 / 10,11,12 / 20,21,22 ...
test_mask = (positions % 10) < 3
train_mask = ~test_mask

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]
y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

numeric_features = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
categorical_features = [c for c in X_train.columns if c not in numeric_features]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

transformers = []
if numeric_features:
    transformers.append(("num", numeric_pipeline, numeric_features))
if categorical_features:
    transformers.append(("cat", categorical_pipeline, categorical_features))

preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression()),
])

try:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
except Exception as e:
    st.error(f"모델 학습 중 오류가 발생했습니다: {e}")
    st.stop()

# 음수 예측도 실제 모델의 결과로 유지하고, 오차 계산에는 그대로 사용
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred) if len(y_test) >= 2 else np.nan

nonzero_actual = y_test != 0
if nonzero_actual.any():
    mape = (
        np.abs(
            (y_test.loc[nonzero_actual].to_numpy() -
             y_pred[nonzero_actual.to_numpy()])
            / y_test.loc[nonzero_actual].to_numpy()
        ).mean() * 100
    )
else:
    mape = np.nan

# 평가 결과 표
result = model_movies.loc[test_mask, ["movieCd", "movieNm", "total_audi"]].copy()
result = result.rename(columns={"total_audi": "실제 총 관객 수"})
result["예측 총 관객 수"] = y_pred
result["절대 오차"] = np.abs(result["실제 총 관객 수"] - result["예측 총 관객 수"])
result["오차율(%)"] = np.where(
    result["실제 총 관객 수"] != 0,
    result["절대 오차"] / np.abs(result["실제 총 관객 수"]) * 100,
    np.nan,
)

# 표시용 반올림
display_result = result.copy()
for col in ["예측 총 관객 수", "절대 오차", "오차율(%)"]:
    display_result[col] = display_result[col].round(2)

# 상단 요약
st.subheader("학습·평가 결과")
c1, c2, c3, c4 = st.columns(4)
c1.metric("학습에 사용한 영화", f"{train_mask.sum():,}편")
c2.metric("평가한 영화", f"{test_mask.sum():,}편")
c3.metric("R² 점수", f"{r2:.4f}" if pd.notna(r2) else "N/A")
c4.metric("MAE", f"{mae:,.0f}명")

st.info(
    f"기준 기간: **{period_text}**  |  "
    f"전체 영화: **{len(movies):,}편**  |  "
    f"선택 변수: **{len(selected_features)}개**"
)

st.write(f"RMSE: **{rmse:,.0f}명**  ·  MAPE: **{mape:.2f}%**" if pd.notna(mape)
         else f"RMSE: **{rmse:,.0f}명**  ·  MAPE: 계산 불가")

st.subheader("테스트용 영화의 실제값 vs 예측값")
st.dataframe(display_result, use_container_width=True, hide_index=True)

# 1,000명 미만 예측 수
low_mask = y_pred < 1000
low_count = int(low_mask.sum())
st.write(f"**예측이 1,000명보다 작은 영화: {low_count}편**")

# 로그 산점도
fig = go.Figure()

# 실제값/예측값이 모두 양수인 일반 점
positive_pred = y_pred > 0
normal_mask = positive_pred & (y_test.to_numpy() > 0) & ~low_mask

fig.add_trace(go.Scatter(
    x=y_test.to_numpy()[normal_mask],
    y=y_pred[normal_mask],
    mode="markers",
    name="테스트 영화",
    text=result.iloc[np.flatnonzero(normal_mask)]["movieNm"].astype(str),
    hovertemplate=(
        "<b>%{text}</b><br>"
        "실제: %{x:,.0f}명<br>"
        "예측: %{y:,.0f}명<extra></extra>"
    ),
))

# 1,000명 미만 예측값은 로그 그래프 바닥(1,000)으로 붙여 표시
low_plot_mask = low_mask & (y_test.to_numpy() > 0)
if low_plot_mask.any():
    fig.add_trace(go.Scatter(
        x=y_test.to_numpy()[low_plot_mask],
        y=np.full(int(low_plot_mask.sum()), 1000.0),
        mode="markers",
        name="예측 < 1,000명",
        text=result.iloc[np.flatnonzero(low_plot_mask)]["movieNm"].astype(str),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "실제: %{x:,.0f}명<br>"
            "실제 예측값: 1,000명 미만<extra></extra>"
        ),
    ))

# 실제값 = 예측값 대각선
positive_actual = y_test.to_numpy()[y_test.to_numpy() > 0]
positive_pred_values = y_pred[y_pred > 0]
positive_values = np.concatenate([positive_actual, positive_pred_values])

if len(positive_values):
    axis_min = max(1, float(np.min(positive_values)) * 0.8)
    axis_max = max(1000, float(np.max(positive_values)) * 1.2)
else:
    axis_min, axis_max = 1, 1000

fig.add_trace(go.Scatter(
    x=[axis_min, axis_max],
    y=[axis_min, axis_max],
    mode="lines",
    name="실제 = 예측",
    hoverinfo="skip",
))

fig.update_layout(
    title="테스트 영화: 실제 총 관객 수 vs 예측 총 관객 수",
    xaxis_title="실제 총 관객 수",
    yaxis_title="예측 총 관객 수",
    template="plotly_white",
    height=650,
    legend_title="구분",
)

fig.update_xaxes(type="log", range=[np.log10(axis_min), np.log10(axis_max)])
fig.update_yaxes(type="log", range=[np.log10(axis_min), np.log10(axis_max)])

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "평가 방식: 영화코드 순 정렬 → 10편마다 앞의 3편을 테스트 → 나머지로 학습. "
    "모델은 선택한 변수로 전처리 후 LinearRegression(다중 선형 회귀)을 사용합니다."
)
