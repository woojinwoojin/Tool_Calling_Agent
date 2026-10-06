import json

import pandas as pd
import pytest

from tools import (
    TOOL_FUNCTIONS,
    TOOLS,
    breakdown,
    compare_periods,
    customer_analysis,
    detect_anomaly,
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
    df["region"] = ["East", "East", "West", "East"]
    return df


@pytest.fixture
def seasonal_df() -> pd.DataFrame:
    """3년치 월별 매출: 매년 11월 피크(계절성) + 해마다 20% 성장 + 2022-06에만 3배 급등."""
    rows = []
    for year in (2021, 2022, 2023):
        for month in range(1, 13):
            sales = 100 * (1.2 ** (year - 2021)) * (2 if month == 11 else 1)
            if (year, month) == (2022, 6):
                sales *= 3
            rows.append((f"{year}-{month:02d}-15", f"O{year}{month}", f"C{month}", "A", sales))
    df = pd.DataFrame(rows, columns=["date", "order_id", "customer_id", "category", "sales"])
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


def test_monthly_trend_filters(df):
    result = monthly_trend(df, "sales", 12, filters={"category": "B"})

    assert result["filters"] == {"category": "B"}
    # B는 1월에만 팔렸지만, 월 축은 전체 데이터 기간(1~4월)으로 유지된다.
    assert [(m["month"], m["value"]) for m in result["months"]] == [
        ("2024-01", 100),
        ("2024-02", 0),
        ("2024-03", 0),
        ("2024-04", 0),
    ]


def test_filters_combine(df):
    # A 카테고리 중 East 지역: O1의 A1(100) + O3의 A1(100)
    result = get_summary(df, filters={"category": "A", "region": "East", "sub_category": None, "segment": None})

    assert result["total_sales"] == 200
    assert result["filters"] == {"category": "A", "region": "East"}  # null 필터는 결과에 표시하지 않는다


def test_filter_value_in_other_column_gets_hint(df):
    result = get_summary(df, filters={"category": "A1"})

    assert result["hint"] == "A1는 sub_category 값입니다."


def test_filter_available_values_respect_earlier_filters(df):
    # B 카테고리 안에는 A1이 없다 → B 안의 하위 카테고리만 보여준다.
    result = discount_impact(df, filters={"category": "B", "sub_category": "A1"})

    assert result["available"] == ["B1"]
    assert "다른 필터와 함께" in result["hint"]


def test_filter_on_missing_column(df):
    result = get_summary(df, filters={"segment": "Consumer"})

    assert "segment" not in result["available"]


def test_breakdown_by_category(df):
    rows = breakdown(df, "category")["rows"]

    assert rows[0] == {"name": "A", "sales": 400, "sales_share_pct": 80.0, "profit": -10, "profit_margin_pct": -2.5}


def test_breakdown_with_filter(df):
    result = breakdown(df, "sub_category", filters={"category": "A"})

    assert result["total"] == {"sales": 400, "profit": -10, "profit_margin_pct": -2.5}
    assert {r["name"] for r in result["rows"]} == {"A1", "A2"}  # 매출이 둘 다 200이라 순서는 보지 않는다
    assert sum(r["sales_share_pct"] for r in result["rows"]) == 100.0


def test_breakdown_by_region(df):
    result = breakdown(df, "region")

    assert [(r["name"], r["sales"]) for r in result["rows"]] == [("East", 300), ("West", 200)]


def test_breakdown_rejects_group_by_used_as_filter(df):
    assert "이미 필터링" in breakdown(df, "category", filters={"category": "A"})["error"]


def test_breakdown_unavailable_group_by(df):
    assert "segment" not in breakdown(df, "segment")["available"]


def test_compare_periods_by_category(df):
    result = compare_periods(df, "2024-01", "2024-01", "2024-02", "2024-02")

    assert result["total"]["sales_change"] == 0  # 200 → 200
    rows = {r["name"]: r for r in result["rows"]}
    assert rows["A"]["sales_change"] == 100
    assert rows["A"]["sales_change_pct"] == 100.0
    assert rows["B"]["compare_sales"] == 0  # 2월에 B 판매 없음 → 0으로 채움
    assert rows["B"]["share_of_total_change_pct"] is None  # 전체 변화가 0이면 비중을 계산할 수 없다


def test_compare_periods_with_filter_orders_by_change(df):
    result = compare_periods(
        df, "2024-02", "2024-02", "2024-04", "2024-04", group_by="sub_category", filters={"category": "A"}
    )

    assert result["total"]["sales_change"] == -100
    # A2: -200, A1: +100 → 변화 크기 순
    assert [(r["name"], r["sales_change"], r["share_of_total_change_pct"]) for r in result["rows"]] == [
        ("A2", -200, 200.0),
        ("A1", 100, -100.0),
    ]
    assert result["rows"][1]["sales_change_pct"] is None  # 기준 기간 매출 0


def test_compare_periods_rejects_same_period(df):
    assert "breakdown" in compare_periods(df, "2024-01", "2024-04", "2024-01", "2024-04")["error"]


def test_compare_periods_empty_period(df):
    assert "기준" in compare_periods(df, "2030-01", "2030-01", "2024-01", "2024-01")["error"]


def test_discount_impact(df):
    bands = {b["discount_band"]: b for b in discount_impact(df)["bands"]}

    assert set(bands) == {"0%", "1~20%", "41%+"}  # 데이터가 없는 구간은 빠진다
    assert bands["0%"]["sales"] == 200
    assert bands["41%+"]["profit_margin_pct"] == -20.0


def test_discount_impact_with_filters(df):
    result = discount_impact(df, filters={"region": "West"})

    assert [b["discount_band"] for b in result["bands"]] == ["41%+"]


def test_discount_impact_without_discount_column(df):
    assert "error" in discount_impact(df.drop(columns="discount"))


def test_customer_analysis_uses_first_purchase_from_all_data(df):
    result = customer_analysis(df, start_month="2024-02", end_month="2024-04")

    # C2는 2월에 처음 구매(신규), C1은 1월에 이미 구매(기존)
    assert result["new_customers"] == 1
    assert result["returning_customers"] == 1
    assert result["new_customer_sales_share_pct"] == round(200 / 300 * 100, 1)
    assert result["repeat_customer_pct"] == 0.0  # 기간 안에서는 둘 다 주문 1번
    assert result["top_10pct_customer_sales_share_pct"] == round(200 / 300 * 100, 1)  # 최소 1명
    assert "note" not in result


def test_customer_analysis_new_means_new_to_company(df):
    # C1은 1월에 West에서 처음 사고, 4월에 East에서 다시 샀다.
    df.loc[df["order_id"] == "O1", "region"] = "West"
    result = customer_analysis(df, filters={"region": "East"}, start_month="2024-02")

    # East에서는 첫 구매지만 회사 전체로는 기존 고객이다.
    assert result["customer_count"] == 1
    assert result["new_customers"] == 0


def test_customer_analysis_whole_period_has_note(df):
    result = customer_analysis(df)

    assert result["new_customers"] == result["customer_count"]
    assert "note" in result


def test_detect_anomaly_ignores_seasonality(seasonal_df):
    anomalies = detect_anomaly(seasonal_df, "sales")["anomalies"]

    # 매년 반복되는 11월 피크는 이상치가 아니고, 2022-06 급등만 잡혀야 한다.
    months = [a["month"] for a in anomalies]
    assert months[0] == "2022-06"
    assert anomalies[0]["deviation_pct"] > 100
    assert not any(m.endswith("-11") for m in months)


def test_detect_anomaly_rejects_profit(df):
    assert "profit" not in detect_anomaly(df, "profit")["available"]


def test_detect_anomaly_needs_two_years(df):
    assert "2년" in detect_anomaly(df, "sales")["error"]


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
