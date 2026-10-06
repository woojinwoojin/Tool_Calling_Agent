import json

import pandas as pd
import pytest

from tools import (
    TOOL_FUNCTIONS,
    TOOLS,
    category_breakdown,
    discount_impact,
    get_summary,
    monthly_trend,
    run_tool,
)


@pytest.fixture
def df() -> pd.DataFrame:
    rows = [
        # (날짜, 주문ID, 고객ID, 카테고리, 하위 카테고리, 매출, 이익, 할인율)
        ("2024-01-10", "O1", "C1", "A", "A1", 100, 20, 0.0),
        ("2024-01-10", "O1", "C1", "B", "B1", 100, 10, 0.2),
        ("2024-02-05", "O2", "C2", "A", "A2", 200, -40, 0.5),
        ("2024-04-01", "O3", "C1", "A", "A1", 100, 10, 0.0),
    ]
    df = pd.DataFrame(
        rows, columns=["date", "order_id", "customer_id", "category", "sub_category", "sales", "profit", "discount"]
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_get_summary(df):
    result = get_summary(df)

    assert result["total_sales"] == 500
    assert result["order_count"] == 3
    assert result["avg_order_value"] == round(500 / 3)
    assert result["profit_margin_pct"] == 0.0


def test_get_summary_filters_period(df):
    result = get_summary(df, start_month="2024-02", end_month="2024-02")

    assert result["period"] == "2024-02 ~ 2024-02"
    assert result["total_sales"] == 200


def test_get_summary_empty_period(df):
    assert "error" in get_summary(df, start_month="2030-01")


def test_monthly_trend_fills_missing_month(df):
    months = monthly_trend(df, "sales", 12)["months"]

    # 3월은 주문이 없지만 0으로 들어가야 4월의 MoM 비교 대상이 맞다.
    assert [m["month"] for m in months] == ["2024-01", "2024-02", "2024-03", "2024-04"]
    assert months[1]["mom_pct"] == 0.0
    assert months[2]["value"] == 0


def test_monthly_trend_keeps_mom_after_slicing(df):
    months = monthly_trend(df, "sales", 1)["months"]

    assert months[0]["month"] == "2024-04"
    assert months[0]["mom_pct"] is None  # 3월 매출 0 → 증가율 계산 불가


def test_monthly_trend_unknown_metric(df):
    result = monthly_trend(df, "revenue", 12)

    assert "error" in result
    assert "sales" in result["available"]


def test_category_breakdown_top_level(df):
    rows = category_breakdown(df)["rows"]

    assert rows[0] == {"name": "A", "sales": 400, "sales_share_pct": 80.0, "profit": -10, "profit_margin_pct": -2.5}


def test_category_breakdown_drills_down(df):
    result = category_breakdown(df, category="A")

    assert result["level"] == "sub_category"
    assert result["total"] == {"sales": 400, "profit": -10, "profit_margin_pct": -2.5}
    assert {r["name"] for r in result["rows"]} == {"A1", "A2"}  # 매출이 둘 다 200이라 순서는 보지 않는다
    assert sum(r["sales_share_pct"] for r in result["rows"]) == 100.0


def test_category_breakdown_unknown_category(df):
    result = category_breakdown(df, category="Z")

    assert result["available"] == ["A", "B"]


def test_discount_impact(df):
    bands = {b["discount_band"]: b for b in discount_impact(df)["bands"]}

    assert set(bands) == {"0%", "1~20%", "41%+"}  # 데이터가 없는 구간은 빠진다
    assert bands["0%"]["sales"] == 200
    assert bands["41%+"]["profit_margin_pct"] == -20.0


def test_discount_impact_without_discount_column(df):
    assert "error" in discount_impact(df.drop(columns="discount"))


def test_run_tool_returns_json(df):
    result = json.loads(run_tool(df, "get_summary", '{"start_month": null, "end_month": null}'))

    assert result["total_sales"] == 500


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("no_such_tool", "{}"),
        ("get_summary", '{"wrong_arg": 1}'),
        ("get_summary", '{"start_month": "2024-13", "end_month": null}'),
    ],
)
def test_run_tool_returns_error_instead_of_raising(df, name, arguments):
    assert "error" in json.loads(run_tool(df, name, arguments))


def test_schemas_match_functions():
    assert [t["name"] for t in TOOLS] == list(TOOL_FUNCTIONS)
    for tool in TOOLS:
        params = tool["parameters"]
        # strict 모드 규칙: 모든 속성이 required에 있어야 한다.
        assert params["required"] == list(params["properties"])
        assert params["additionalProperties"] is False
