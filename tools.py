"""LLM이 호출할 수 있는 분석 도구(tool)와 그 스키마.

각 도구는 역할 이름 컬럼(date, sales, ...)의 df를 받아 JSON으로 바꿀 수 있는 dict를 돌려준다.
LLM에는 원본 행이 아니라 이 요약 결과만 전달된다.
잘못된 인자는 예외 대신 {"error": ...}로 돌려줘서, LLM이 결과를 보고 인자를 고쳐 다시 호출할 수 있게 한다.
"""

import json
from collections.abc import Callable

import pandas as pd

METRICS = ["sales", "profit", "orders", "customers"]
MAX_MONTHS = 36

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


# ---------------------------------------------------------------- 도구


def get_summary(df: pd.DataFrame, start_month: str | None = None, end_month: str | None = None) -> dict:
    df = filter_period(df, start_month, end_month)
    if df.empty:
        return {"error": "해당 기간에 데이터가 없습니다."}

    orders = df["order_id"].nunique()
    result = {
        "period": period_label(df),
        "total_sales": round(df["sales"].sum()),
        "order_count": orders,
        "customer_count": df["customer_id"].nunique(),
        "avg_order_value": round(df["sales"].sum() / orders),
    }
    if "profit" in df:
        result["total_profit"] = round(df["profit"].sum())
        result["profit_margin_pct"] = pct(df["profit"].sum() / df["sales"].sum())
    return result


def monthly_trend(df: pd.DataFrame, metric: str = "sales", last_n_months: int = 12) -> dict:
    if metric not in METRICS or (metric == "profit" and "profit" not in df):
        available = [m for m in METRICS if m != "profit" or "profit" in df]
        return {"error": f"지원하지 않는 metric: {metric}", "available": available}

    aggs = {
        "sales": ("sales", "sum"),
        "orders": ("order_id", "nunique"),
        "customers": ("customer_id", "nunique"),
    }
    if "profit" in df:
        aggs["profit"] = ("profit", "sum")
    monthly = df.groupby(df["date"].dt.to_period("M")).agg(**aggs)[metric]

    # 주문이 없는 달도 0으로 채워야 MoM/YoY 비교 대상이 어긋나지 않는다.
    full_range = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(full_range, fill_value=0)

    # MoM/YoY는 전체 기간으로 계산한 뒤 자른다. 먼저 자르면 첫 달들의 비교 대상이 사라진다.
    table = pd.DataFrame({"value": monthly, "mom": monthly.pct_change(), "yoy": monthly.pct_change(periods=12)})
    n = max(1, min(last_n_months, MAX_MONTHS))
    return {
        "metric": metric,
        "months": [
            {"month": str(month), "value": round(row["value"]), "mom_pct": pct(row["mom"]), "yoy_pct": pct(row["yoy"])}
            for month, row in table.tail(n).iterrows()
        ],
    }


def category_breakdown(
    df: pd.DataFrame,
    category: str | None = None,
    start_month: str | None = None,
    end_month: str | None = None,
) -> dict:
    df = filter_period(df, start_month, end_month)
    if df.empty:
        return {"error": "해당 기간에 데이터가 없습니다."}

    value_cols = ["sales", "profit"] if "profit" in df else ["sales"]
    if category is None:
        grouped = df.groupby("category")[value_cols].sum()
        return {
            "period": period_label(df),
            "level": "category",
            "total": sales_profit_total(df),
            "rows": sales_profit_rows(grouped, df["sales"].sum()),
        }

    categories = sorted(df["category"].unique())
    if category not in categories:
        return {"error": f"없는 카테고리: {category}", "available": categories}
    if "sub_category" not in df:
        return {"error": "이 데이터에는 하위 카테고리 컬럼이 없습니다."}

    subset = df[df["category"] == category]
    grouped = subset.groupby("sub_category")[value_cols].sum()
    return {
        "period": period_label(df),
        "level": "sub_category",
        "category": category,
        "total": sales_profit_total(subset),
        "rows": sales_profit_rows(grouped, subset["sales"].sum()),
    }


def discount_impact(df: pd.DataFrame, category: str | None = None) -> dict:
    if "discount" not in df or "profit" not in df:
        return {"error": "이 데이터에는 할인율 또는 이익 컬럼이 없습니다."}
    if category is not None:
        categories = sorted(df["category"].unique())
        if category not in categories:
            return {"error": f"없는 카테고리: {category}", "available": categories}
        df = df[df["category"] == category]

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
    return {"category": category or "전체", "bands": rows}


# ---------------------------------------------------------------- 스키마

# OpenAI Responses API의 function tool 형식.
# strict 모드에서는 모든 속성을 required에 넣어야 하므로, 선택 인자는 null을 허용하는 타입으로 표현한다.
MONTH = {"type": ["string", "null"], "description": "'YYYY-MM' 형식. 제한하지 않으려면 null."}
CATEGORY = {"type": ["string", "null"], "description": "카테고리 이름 (예: Furniture). 전체를 보려면 null."}


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
        "기간 전체의 총 매출, 총 이익, 이익률, 주문 수, 고객 수, 평균 주문금액을 계산한다.",
        {"start_month": MONTH, "end_month": MONTH},
    ),
    function_tool(
        "monthly_trend",
        "지표 하나의 월별 값과 MoM(전월 대비), YoY(전년 동월 대비) 변화율을 최근 N개월만큼 돌려준다. "
        "매출 감소/증가의 시점과 계절성을 확인할 때 쓴다.",
        {
            "metric": {"type": "string", "enum": METRICS, "description": "볼 지표"},
            "last_n_months": {"type": "integer", "description": f"최근 몇 개월 (1~{MAX_MONTHS})"},
        },
    ),
    function_tool(
        "category_breakdown",
        "category가 null이면 카테고리별, 값이 있으면 그 카테고리의 하위 카테고리별로 "
        "매출, 매출 비중, 이익, 이익률을 계산한다. 어느 부문이 원인인지 좁혀갈 때 쓴다.",
        {"category": CATEGORY, "start_month": MONTH, "end_month": MONTH},
    ),
    function_tool(
        "discount_impact",
        "할인율 구간(0%, 1~20%, 21~40%, 41%+)별 매출, 이익, 이익률을 계산한다. "
        "이익률이 낮은 원인이 할인인지 확인할 때 쓴다.",
        {"category": CATEGORY},
    ),
]

TOOL_FUNCTIONS: dict[str, Callable[..., dict]] = {
    "get_summary": get_summary,
    "monthly_trend": monthly_trend,
    "category_breakdown": category_breakdown,
    "discount_impact": discount_impact,
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
