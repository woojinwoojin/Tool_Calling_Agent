import pandas as pd
import pytest

from analysis import ColumnMap, Dataset, category_kpis, load_data, monthly_kpis, summarize


def make_df(rows: list[tuple]) -> pd.DataFrame:
    """(날짜, 주문ID, 고객ID, 카테고리, 매출, 이익) 튜플로 역할 이름 컬럼의 df를 만든다."""
    df = pd.DataFrame(rows, columns=["date", "order_id", "customer_id", "category", "sales", "profit"])
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return make_df(
        [
            # 주문 O1은 상품 2개(2행) — 주문 수는 1로 세야 한다
            ("2024-01-10", "O1", "C1", "A", 100, 10),
            ("2024-01-10", "O1", "C1", "B", 50, 5),
            ("2024-01-20", "O2", "C2", "A", 50, -5),
            ("2024-02-05", "O3", "C1", "B", 300, 30),
        ]
    )


def test_summarize_counts_unique_orders_and_customers(sample_df):
    summary = summarize(sample_df)

    assert summary["total_sales"] == 500
    assert summary["total_profit"] == 40
    assert summary["order_count"] == 3  # 행은 4개지만 주문은 3건
    assert summary["customer_count"] == 2


def test_summarize_without_profit(sample_df):
    summary = summarize(sample_df.drop(columns="profit"))

    assert "total_profit" not in summary


def test_category_kpis_profit_margin(sample_df):
    result = category_kpis(sample_df)

    assert list(result.index) == ["B", "A"]  # 매출 큰 순서
    assert result.loc["A", "sales"] == 150
    assert result.loc["A", "profit_margin"] == pytest.approx(5 / 150)


def test_monthly_kpis_mom(sample_df):
    monthly = monthly_kpis(sample_df)

    assert monthly.loc["2024-01", "sales"] == 200
    assert monthly.loc["2024-01", "orders"] == 2
    assert monthly.loc["2024-01", "customers"] == 2
    assert pd.isna(monthly.loc["2024-01", "sales_mom"])  # 비교할 전월 없음
    assert monthly.loc["2024-02", "sales_mom"] == pytest.approx(0.5)  # 200 → 300


def test_monthly_kpis_fills_missing_months():
    df = make_df(
        [
            ("2024-01-15", "O1", "C1", "A", 100, 0),
            ("2024-03-15", "O2", "C1", "A", 100, 0),  # 2월은 주문 없음
        ]
    )

    monthly = monthly_kpis(df)

    assert [str(m) for m in monthly.index] == ["2024-01", "2024-02", "2024-03"]
    assert monthly.loc["2024-02", "sales"] == 0


def test_monthly_kpis_yoy_compares_same_month_last_year():
    df = make_df(
        [
            ("2023-01-15", "O1", "C1", "A", 100, 0),
            ("2023-12-15", "O2", "C1", "A", 999, 0),
            ("2024-01-15", "O3", "C1", "A", 150, 0),
        ]
    )

    monthly = monthly_kpis(df)

    # 빠진 달(2023-02 ~ 11)이 채워져 있어야 12칸 전이 정확히 작년 1월이 된다
    assert monthly.loc["2024-01", "sales_yoy"] == pytest.approx(0.5)  # 100 → 150


def test_load_data_renames_columns_by_mapping(tmp_path):
    csv = tmp_path / "shop.csv"
    csv.write_text(
        "주문일,주문번호,회원,분류,금액,메모\n"
        "2024-01-10,O1,C1,A,100,무시되는 컬럼\n",
        encoding="utf-8",
    )
    dataset = Dataset(
        path=csv,
        columns=ColumnMap(date="주문일", sales="금액", order_id="주문번호", customer_id="회원", category="분류"),
        currency="KRW",
    )

    df = load_data(dataset)

    assert set(df.columns) == {"date", "sales", "order_id", "customer_id", "category"}
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
