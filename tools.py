"""LLM이 호출할 수 있는 분석 도구(tool)와 그 스키마.

각 도구는 역할 이름 컬럼(date, sales, ...)의 df를 받아 JSON으로 바꿀 수 있는 dict를 돌려준다.
LLM에는 원본 행이 아니라 이 요약 결과만 전달된다.
잘못된 인자는 예외 대신 {"error": ...}로 돌려줘서, LLM이 결과를 보고 인자를 고쳐 다시 호출할 수 있게 한다.

인자는 두 종류로 나눈다.
- filters: 어떤 행만 볼지 (모든 도구 공통). 예: {"region": "Central", "category": "Furniture"}
- group_by: 무엇으로 나눠 볼지 (나눠 보는 도구만). 예: "sub_category"
"""

import json
from collections.abc import Callable

import pandas as pd

METRICS = ["sales", "profit", "orders", "customers"]
ANOMALY_METRICS = ["sales", "orders", "customers"]
# 필터와 group_by에 쓸 수 있는 역할. 데이터에 없는 역할은 쓰면 error를 돌려준다.
DIMENSIONS = ["category", "sub_category", "region", "segment"]
MAX_MONTHS = 36
TOP_CUSTOMER_RATIO = 0.1  # 상위 고객 매출 비중을 볼 때의 상위 비율
ANOMALY_STD = 2  # 기대값 대비 편차가 표준편차의 몇 배를 넘으면 이상치로 볼지
MAX_ANOMALIES = 10

# 할인율 구간: (이름, 최소 초과, 최대 이하). 0%는 따로 본다.
DISCOUNT_BANDS = [
    ("0%", -1.0, 0.0),
    ("1~20%", 0.0, 0.2),
    ("21~40%", 0.2, 0.4),
    ("41%+", 0.4, 1.0),
]


def pct(value: float) -> float | None:
    """비율(0.123)을 퍼센트(12.3)로 바꾼다. 계산할 수 없으면 None."""
    if pd.isna(value) or value in (float("inf"), float("-inf")):
        return None
    return round(float(value) * 100, 1)


def change_pct(new: float, old: float) -> float | None:
    return pct((new - old) / old) if old else None


def filter_period(df: pd.DataFrame, start_month: str | None, end_month: str | None) -> pd.DataFrame:
    """'YYYY-MM' 형식의 시작/끝 월(포함)로 행을 거른다. None이면 그쪽은 제한하지 않는다."""
    month = df["date"].dt.to_period("M")
    if start_month:
        df = df[month >= pd.Period(start_month, freq="M")]
        month = month[df.index]
    if end_month:
        df = df[month <= pd.Period(end_month, freq="M")]
    return df


def period_label(df: pd.DataFrame) -> str:
    return f"{df['date'].min():%Y-%m} ~ {df['date'].max():%Y-%m}"


def active_filters(filters: dict | None) -> dict:
    """결과에 표시할, 값이 있는 필터만. LLM이 어떤 범위의 숫자인지 헷갈리지 않게 결과마다 붙인다."""
    return {key: value for key, value in (filters or {}).items() if value is not None}


def apply_filters(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame | dict:
    """filters의 값이 있는 항목으로 행을 거른다. 잘못된 필터면 LLM이 고칠 수 있는 error dict를 돌려준다."""
    full = df
    for key, value in active_filters(filters).items():
        if key not in DIMENSIONS or key not in df:
            return {"error": f"이 데이터에서 쓸 수 없는 필터: {key}", "available": [d for d in DIMENSIONS if d in df]}
        # 앞의 필터로 이미 걸러진 범위 안에서 고를 수 있는 값을 보여준다 (예: Furniture 안의 하위 카테고리).
        values = sorted(df[key].unique())
        if value in values:
            df = df[df[key] == value]
            continue

        error = {"error": f"{key}에 없는 값: {value}", "available": values}
        other = [d for d in DIMENSIONS if d != key and d in full and value in full[d].values]
        if other:
            error["hint"] = f"{value}는 {other[0]} 값입니다."
        elif value in full[key].values:
            error["hint"] = "값은 있지만 다른 필터와 함께 쓰면 해당하는 행이 없습니다."
        return error
    return df


def check_group_by(df: pd.DataFrame, group_by: str, filters: dict | None) -> dict | None:
    if group_by not in DIMENSIONS or group_by not in df:
        return {"error": f"이 데이터에서 나눠 볼 수 없는 기준: {group_by}", "available": [d for d in DIMENSIONS if d in df]}
    if group_by in active_filters(filters):
        return {"error": f"{group_by}로 이미 필터링했습니다. 다른 group_by를 고르세요."}
    return None


def value_columns(df: pd.DataFrame) -> list[str]:
    return ["sales", "profit"] if "profit" in df else ["sales"]


def sales_profit_total(df: pd.DataFrame) -> dict:
    """행 목록의 합계. LLM이 행을 직접 더해 합계를 계산하지 않도록 함께 돌려준다."""
    total = {"sales": round(df["sales"].sum())}
    if "profit" in df:
        total["profit"] = round(df["profit"].sum())
        total["profit_margin_pct"] = pct(df["profit"].sum() / df["sales"].sum())
    return total


def sales_profit_rows(grouped: pd.DataFrame, total_sales: float) -> list[dict]:
    """groupby 결과(sales, profit 컬럼)를 매출 순 행 목록으로 바꾼다."""
    rows = []
    for name, row in grouped.sort_values("sales", ascending=False).iterrows():
        item = {"name": str(name), "sales": round(row["sales"]), "sales_share_pct": pct(row["sales"] / total_sales)}
        if "profit" in row:
            item["profit"] = round(row["profit"])
            item["profit_margin_pct"] = pct(row["profit"] / row["sales"])
        rows.append(item)
    return rows


def change_row(name: str, base: pd.Series, compare: pd.Series, total_change: float) -> dict:
    """두 기간의 sales(, profit) 합계로 증감 행을 만든다."""
    sales_change = compare["sales"] - base["sales"]
    row = {
        "name": name,
        "base_sales": round(base["sales"]),
        "compare_sales": round(compare["sales"]),
        "sales_change": round(sales_change),
        "sales_change_pct": change_pct(compare["sales"], base["sales"]),
        # 전체 매출 변화 중 이 항목이 차지하는 비중. "감소분의 대부분"을 LLM이 직접 계산하지 않게 한다.
        "share_of_total_change_pct": pct(sales_change / total_change) if total_change else None,
    }
    if "profit" in base:
        row["base_profit"] = round(base["profit"])
        row["compare_profit"] = round(compare["profit"])
        row["profit_change"] = round(compare["profit"] - base["profit"])
    return row


def monthly_series(df: pd.DataFrame, metric: str, filters: dict | None) -> pd.Series | dict:
    """지표 하나의 월별 값. 인자가 잘못되면 error dict를 돌려준다."""
    if metric not in METRICS or (metric == "profit" and "profit" not in df):
        available = [m for m in METRICS if m != "profit" or "profit" in df]
        return {"error": f"지원하지 않는 metric: {metric}", "available": available}

    # 월 축은 필터로 거르기 전의 전체 기간으로 잡는다. 거른 범위에 판매가 없는 달도 0으로 보여야 한다.
    month = df["date"].dt.to_period("M")
    full_range = pd.period_range(month.min(), month.max(), freq="M")
    df = apply_filters(df, filters)
    if isinstance(df, dict):
        return df

    aggs = {
        "sales": ("sales", "sum"),
        "orders": ("order_id", "nunique"),
        "customers": ("customer_id", "nunique"),
    }
    if "profit" in df:
        aggs["profit"] = ("profit", "sum")
    monthly = df.groupby(df["date"].dt.to_period("M")).agg(**aggs)[metric]

    # 주문이 없는 달도 0으로 채워야 MoM/YoY 비교 대상이 어긋나지 않는다.
    return monthly.reindex(full_range, fill_value=0)


# ---------------------------------------------------------------- 도구


def get_summary(
    df: pd.DataFrame, filters: dict | None = None, start_month: str | None = None, end_month: str | None = None
) -> dict:
    df = apply_filters(df, filters)
    if isinstance(df, dict):
        return df
    df = filter_period(df, start_month, end_month)
    if df.empty:
        return {"error": "해당 기간에 데이터가 없습니다."}

    orders = df["order_id"].nunique()
    result = {
        "period": period_label(df),
        "filters": active_filters(filters),
        "total_sales": round(df["sales"].sum()),
        "order_count": orders,
        "customer_count": df["customer_id"].nunique(),
        "avg_order_value": round(df["sales"].sum() / orders),
    }
    if "profit" in df:
        result["total_profit"] = round(df["profit"].sum())
        result["profit_margin_pct"] = pct(df["profit"].sum() / df["sales"].sum())
    return result


def monthly_trend(
    df: pd.DataFrame, metric: str = "sales", last_n_months: int = 12, filters: dict | None = None
) -> dict:
    monthly = monthly_series(df, metric, filters)
    if isinstance(monthly, dict):
        return monthly

    # MoM/YoY는 전체 기간으로 계산한 뒤 자른다. 먼저 자르면 첫 달들의 비교 대상이 사라진다.
    table = pd.DataFrame({"value": monthly, "mom": monthly.pct_change(), "yoy": monthly.pct_change(periods=12)})
    n = max(1, min(last_n_months, MAX_MONTHS))
    return {
        "metric": metric,
        "filters": active_filters(filters),
        "months": [
            {"month": str(month), "value": round(row["value"]), "mom_pct": pct(row["mom"]), "yoy_pct": pct(row["yoy"])}
            for month, row in table.tail(n).iterrows()
        ],
    }


def breakdown(
    df: pd.DataFrame,
    group_by: str,
    filters: dict | None = None,
    start_month: str | None = None,
    end_month: str | None = None,
) -> dict:
    if error := check_group_by(df, group_by, filters):
        return error
    df = apply_filters(df, filters)
    if isinstance(df, dict):
        return df
    df = filter_period(df, start_month, end_month)
    if df.empty:
        return {"error": "해당 기간에 데이터가 없습니다."}

    grouped = df.groupby(group_by)[value_columns(df)].sum()
    return {
        "period": period_label(df),
        "group_by": group_by,
        "filters": active_filters(filters),
        "total": sales_profit_total(df),
        "rows": sales_profit_rows(grouped, df["sales"].sum()),
    }


def compare_periods(
    df: pd.DataFrame,
    base_start_month: str,
    base_end_month: str,
    compare_start_month: str,
    compare_end_month: str,
    group_by: str = "category",
    filters: dict | None = None,
) -> dict:
    if (base_start_month, base_end_month) == (compare_start_month, compare_end_month):
        return {"error": "기준 기간과 비교 기간이 같습니다. 한 기간만 보려면 breakdown을 쓰세요."}
    if error := check_group_by(df, group_by, filters):
        return error
    df = apply_filters(df, filters)
    if isinstance(df, dict):
        return df

    base = filter_period(df, base_start_month, base_end_month)
    compare = filter_period(df, compare_start_month, compare_end_month)
    if base.empty or compare.empty:
        return {"error": f"{'기준' if base.empty else '비교'} 기간에 데이터가 없습니다."}

    cols = value_columns(df)
    table = pd.concat(
        {"base": base.groupby(group_by)[cols].sum(), "compare": compare.groupby(group_by)[cols].sum()},
        axis=1,
    ).fillna(0)  # 한쪽 기간에만 판매된 항목은 다른 쪽을 0으로 본다
    base_total, compare_total = table["base"].sum(), table["compare"].sum()
    total_change = compare_total["sales"] - base_total["sales"]

    # 변화가 큰 항목(증가든 감소든)부터 보여준다.
    order = (table["compare"]["sales"] - table["base"]["sales"]).abs().sort_values(ascending=False).index
    return {
        "base_period": f"{base_start_month} ~ {base_end_month}",
        "compare_period": f"{compare_start_month} ~ {compare_end_month}",
        "group_by": group_by,
        "filters": active_filters(filters),
        "total": change_row("total", base_total, compare_total, total_change),
        "rows": [change_row(str(name), table["base"].loc[name], table["compare"].loc[name], total_change) for name in order],
    }


def discount_impact(df: pd.DataFrame, filters: dict | None = None) -> dict:
    if "discount" not in df or "profit" not in df:
        return {"error": "이 데이터에는 할인율 또는 이익 컬럼이 없습니다."}
    df = apply_filters(df, filters)
    if isinstance(df, dict):
        return df

    rows = []
    for name, low, high in DISCOUNT_BANDS:
        band = df[(df["discount"] > low) & (df["discount"] <= high)]
        if band.empty:
            continue
        rows.append(
            {
                "discount_band": name,
                "line_items": len(band),
                "sales": round(band["sales"].sum()),
                "profit": round(band["profit"].sum()),
                "profit_margin_pct": pct(band["profit"].sum() / band["sales"].sum()),
            }
        )
    return {"filters": active_filters(filters), "bands": rows}


def customer_analysis(
    df: pd.DataFrame, filters: dict | None = None, start_month: str | None = None, end_month: str | None = None
) -> dict:
    # 신규 여부는 필터와 기간으로 거르기 전, 회사 전체 데이터에서의 첫 구매 월로 판단한다.
    # (Central 필터를 걸어도 다른 지역에서 산 적이 있는 고객은 신규가 아니다.)
    first_month = df.groupby("customer_id")["date"].min().dt.to_period("M")
    period = apply_filters(df, filters)
    if isinstance(period, dict):
        return period
    period = filter_period(period, start_month, end_month)
    if period.empty:
        return {"error": "해당 기간에 데이터가 없습니다."}

    start = period["date"].min().to_period("M")
    customers = period["customer_id"].unique()
    new_customers = set(first_month[customers][first_month[customers] >= start].index)
    orders_per_customer = period.groupby("customer_id")["order_id"].nunique()
    sales_per_customer = period.groupby("customer_id")["sales"].sum().sort_values(ascending=False)
    top_n = max(1, round(len(customers) * TOP_CUSTOMER_RATIO))
    total_sales = period["sales"].sum()

    result = {
        "period": period_label(period),
        "filters": active_filters(filters),
        "customer_count": len(customers),
        "new_customers": len(new_customers),
        "returning_customers": len(customers) - len(new_customers),
        "new_customer_sales_share_pct": pct(
            period.loc[period["customer_id"].isin(new_customers), "sales"].sum() / total_sales
        ),
        # 기간 안에서 주문을 2번 이상 한 고객 비율
        "repeat_customer_pct": pct((orders_per_customer >= 2).mean()),
        "avg_orders_per_customer": round(float(orders_per_customer.mean()), 2),
        "avg_sales_per_customer": round(total_sales / len(customers)),
        f"top_{round(TOP_CUSTOMER_RATIO * 100)}pct_customer_sales_share_pct": pct(
            sales_per_customer.head(top_n).sum() / total_sales
        ),
    }
    if start == first_month.min():
        result["note"] = "기간이 데이터 시작 월부터라서 모든 고객이 신규로 집계됩니다."
    return result


def detect_anomaly(df: pd.DataFrame, metric: str = "sales", filters: dict | None = None) -> dict:
    # 이익은 0 근처나 음수인 달이 있어 '기대값 대비 몇 %'가 수천 %로 튄다. 항상 양수인 지표만 받는다.
    if metric not in ANOMALY_METRICS:
        return {
            "error": f"이상 탐지를 지원하지 않는 metric: {metric}",
            "available": ANOMALY_METRICS,
            "hint": "이익은 0 근처나 음수인 달이 있어 비율 편차를 계산할 수 없습니다. monthly_trend로 확인하세요.",
        }
    monthly = monthly_series(df, metric, filters)
    if isinstance(monthly, dict):
        return monthly
    if monthly.index.year.nunique() < 2:
        return {"error": "계절성을 계산하려면 2년 이상의 데이터가 필요합니다."}

    # 기대값 = 그해 월평균 × 그 달의 계절 지수.
    # 계절 지수는 '다른 해'의 같은 달 값으로만 계산한다. 자기 자신을 넣으면 이상치가 기대값을 끌어올려 덜 튀어 보인다.
    # 한계: 마지막 해가 일부 월만 있으면 그해 월평균이 계절성 때문에 치우칠 수 있다.
    year = pd.Series(monthly.index.year, index=monthly.index)
    calendar_month = pd.Series(monthly.index.month, index=monthly.index)
    year_mean = monthly.groupby(year).transform("mean")
    index = monthly / year_mean
    same_month = index.groupby(calendar_month)
    seasonal = (same_month.transform("sum") - index) / (same_month.transform("count") - 1)
    expected = year_mean * seasonal
    deviation = (monthly / expected - 1).replace([float("inf"), float("-inf")], float("nan"))

    threshold = ANOMALY_STD * deviation.std()
    flagged = deviation[deviation.abs() > threshold].sort_values(key=abs, ascending=False)
    return {
        "metric": metric,
        "filters": active_filters(filters),
        "method": f"계절성과 연간 수준을 반영한 기대값 대비 편차가 표준편차의 {ANOMALY_STD}배를 넘는 달",
        "threshold_pct": pct(threshold),
        "anomalies": [
            {
                "month": str(month),
                "value": round(monthly[month]),
                "expected": round(expected[month]),
                "deviation_pct": pct(deviation[month]),
            }
            for month in flagged.index[:MAX_ANOMALIES]
        ],
    }


# ---------------------------------------------------------------- 스키마

# OpenAI Responses API의 function tool 형식.
# strict 모드에서는 모든 속성을 required에 넣어야 하므로, 선택 인자는 null을 허용하는 타입으로 표현한다.
MONTH = {"type": ["string", "null"], "description": "'YYYY-MM' 형식. 제한하지 않으려면 null."}
REQUIRED_MONTH = {"type": "string", "description": "'YYYY-MM' 형식"}
FILTERS = {
    "type": "object",
    "description": "어떤 행만 볼지. 거르지 않을 항목은 null. 여러 항목을 함께 쓰면 모두 만족하는 행만 남는다.",
    "properties": {
        "category": {"type": ["string", "null"], "description": "카테고리 (예: Furniture)"},
        "sub_category": {"type": ["string", "null"], "description": "하위 카테고리 (예: Chairs)"},
        "region": {"type": ["string", "null"], "description": "지역 (예: Central)"},
        "segment": {"type": ["string", "null"], "description": "고객 세그먼트 (예: Consumer)"},
    },
    "required": DIMENSIONS,
    "additionalProperties": False,
}
GROUP_BY = {"type": "string", "enum": DIMENSIONS, "description": "무엇으로 나눠 볼지. filters에 쓴 항목은 고를 수 없다."}


def function_tool(name: str, description: str, properties: dict) -> dict:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
        "strict": True,
    }


TOOLS = [
    function_tool(
        "get_summary",
        "총 매출, 총 이익, 이익률, 주문 수, 고객 수, 평균 주문금액을 계산한다.",
        {"filters": FILTERS, "start_month": MONTH, "end_month": MONTH},
    ),
    function_tool(
        "monthly_trend",
        "지표 하나의 월별 값과 MoM(전월 대비), YoY(전년 동월 대비) 변화율을 최근 N개월만큼 돌려준다. "
        "매출 변화의 시점과 계절성을 확인할 때 쓴다.",
        {
            "metric": {"type": "string", "enum": METRICS, "description": "볼 지표"},
            "last_n_months": {"type": "integer", "description": f"최근 몇 개월 (1~{MAX_MONTHS})"},
            "filters": FILTERS,
        },
    ),
    function_tool(
        "breakdown",
        "group_by 기준(카테고리, 하위 카테고리, 지역, 고객 세그먼트)별 매출, 매출 비중, 이익, 이익률과 합계를 계산한다. "
        "어느 부문이 원인인지 좁혀갈 때 쓴다. 예: Central 지역의 카테고리별 → "
        'group_by="category", filters.region="Central"',
        {"group_by": GROUP_BY, "filters": FILTERS, "start_month": MONTH, "end_month": MONTH},
    ),
    function_tool(
        "compare_periods",
        "두 기간(기준 vs 비교)의 매출, 이익과 증감액, 증감률, 전체 변화 중 비중을 group_by 항목별로 계산한다. "
        "'왜 매출이 떨어졌/늘었나' 같은 변화 원인 질문에 쓴다.",
        {
            "base_start_month": REQUIRED_MONTH,
            "base_end_month": REQUIRED_MONTH,
            "compare_start_month": REQUIRED_MONTH,
            "compare_end_month": REQUIRED_MONTH,
            "group_by": GROUP_BY,
            "filters": FILTERS,
        },
    ),
    function_tool(
        "discount_impact",
        "할인율 구간(0%, 1~20%, 21~40%, 41%+)별 매출, 이익, 이익률을 계산한다. "
        "이익률이 낮은 원인이 할인인지 확인할 때 쓴다.",
        {"filters": FILTERS},
    ),
    function_tool(
        "customer_analysis",
        "기간 내 고객 수, 신규/기존 고객 수와 신규 고객 매출 비중, 재구매 고객 비율(기간 내 주문 2회 이상), "
        f"고객당 평균 주문 수와 매출, 상위 {round(TOP_CUSTOMER_RATIO * 100)}% 고객의 매출 비중을 계산한다. "
        "신규 여부는 필터와 상관없이 회사 전체 데이터에서의 첫 구매 월로 판단한다. "
        "재구매율은 기간이 길수록 높아지므로, 일반적인 수준을 볼 때는 1년 단위로 보는 것이 좋다.",
        {"filters": FILTERS, "start_month": MONTH, "end_month": MONTH},
    ),
    function_tool(
        "detect_anomaly",
        "전체 기간에서 계절성과 연간 수준으로 예상한 값보다 크게 높거나 낮았던 달을 찾는다. "
        "MoM만으로는 매년 반복되는 성수기/비수기도 급등락처럼 보이므로, 특이한 달을 찾을 때는 이 도구를 쓴다.",
        {
            "metric": {"type": "string", "enum": ANOMALY_METRICS, "description": "볼 지표 (이익은 지원하지 않음)"},
            "filters": FILTERS,
        },
    ),
]

TOOL_FUNCTIONS: dict[str, Callable[..., dict]] = {
    "get_summary": get_summary,
    "monthly_trend": monthly_trend,
    "breakdown": breakdown,
    "compare_periods": compare_periods,
    "discount_impact": discount_impact,
    "customer_analysis": customer_analysis,
    "detect_anomaly": detect_anomaly,
}


def run_tool(df: pd.DataFrame, name: str, arguments: str) -> str:
    """LLM이 요청한 도구를 실행하고, 결과를 LLM에 돌려줄 JSON 문자열로 만든다."""
    if name not in TOOL_FUNCTIONS:
        result = {"error": f"없는 도구: {name}", "available": list(TOOL_FUNCTIONS)}
    else:
        try:
            result = TOOL_FUNCTIONS[name](df, **json.loads(arguments))
        except (TypeError, ValueError) as e:  # 인자 이름/형식 오류 (예: 잘못된 월 형식)
            result = {"error": f"인자 오류: {e}"}
    return json.dumps(result, ensure_ascii=False)
