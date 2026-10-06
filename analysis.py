"""CSV 데이터에서 1주차 KPI를 계산한다.

분석 함수는 실제 컬럼 이름 대신 '역할(role)' 이름만 사용한다.
데이터셋마다 다른 컬럼 이름은 ColumnMap으로 역할에 연결한다.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd


@dataclass
class ColumnMap:
    date: str  # 주문 날짜
    sales: str  # 매출 금액
    order_id: str  # 주문 ID (한 주문이 여러 행일 수 있음)
    customer_id: str  # 고객 ID
    category: str  # 카테고리
    profit: str | None = None  # 이익 (없는 데이터셋도 있음)
    sub_category: str | None = None  # 하위 카테고리
    discount: str | None = None  # 할인율 (0.2 = 20%)
    region: str | None = None  # 지역
    segment: str | None = None  # 고객 세그먼트


@dataclass
class Dataset:
    path: Path | BinaryIO  # 파일 경로 또는 업로드된 파일 객체
    columns: ColumnMap
    currency: str  # 금액 단위 (보고서에 표시)
    encoding: str = "utf-8"
    date_format: str | None = None


# 2주차에는 이 매핑을 LLM이 컬럼 이름 + 샘플 행을 보고 추론하게 만든다.
SUPERSTORE = Dataset(
    path=Path(__file__).parent / "data" / "Sample - Superstore.csv",
    columns=ColumnMap(
        date="Order Date",
        sales="Sales",
        order_id="Order ID",
        customer_id="Customer ID",
        category="Category",
        profit="Profit",
        sub_category="Sub-Category",
        discount="Discount",
        region="Region",
        segment="Segment",
    ),
    currency="USD",
    # 이 파일은 UTF-8이 아니라 cp1252이고, 날짜는 월/일/년 형식이다.
    encoding="cp1252",
    date_format="%m/%d/%Y",
)


def load_data(dataset: Dataset) -> pd.DataFrame:
    """CSV를 읽고 컬럼 이름을 역할 이름으로 바꾼다."""
    cols = dataset.columns
    rename = {
        getattr(cols, role): role
        for role in vars(cols)
        if getattr(cols, role) is not None
    }
    df = pd.read_csv(dataset.path, encoding=dataset.encoding, usecols=list(rename))
    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"], format=dataset.date_format)
    return df


def summarize(df: pd.DataFrame) -> dict:
    # 한 행은 '주문'이 아니라 '주문 안의 상품 1개'일 수 있으므로 주문/고객 수는 고유값으로 센다.
    summary = {
        "total_sales": df["sales"].sum(),
        "order_count": df["order_id"].nunique(),
        "customer_count": df["customer_id"].nunique(),
    }
    if "profit" in df:
        summary["total_profit"] = df["profit"].sum()
    return summary


def sales_by_category(df: pd.DataFrame) -> pd.Series:
    return df.groupby("category")["sales"].sum().sort_values(ascending=False)


def category_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """카테고리별 매출, 이익, 이익률(이익 데이터가 있을 때)."""
    if "profit" not in df:
        return sales_by_category(df).to_frame()
    result = df.groupby("category")[["sales", "profit"]].sum()
    result["profit_margin"] = result["profit"] / result["sales"]
    return result.sort_values("sales", ascending=False)


def monthly_kpis(df: pd.DataFrame) -> pd.DataFrame:
    aggs = {
        "sales": ("sales", "sum"),
        "orders": ("order_id", "nunique"),
        "customers": ("customer_id", "nunique"),
    }
    if "profit" in df:
        aggs["profit"] = ("profit", "sum")
    monthly = df.groupby(df["date"].dt.to_period("M")).agg(**aggs)

    # 주문이 없는 달이 있으면 비교 대상이 어긋나므로, 빠진 달을 0으로 채워 연속된 월 축을 만든다.
    full_range = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(full_range, fill_value=0)
    monthly.index.name = "month"

    # MoM: 직전 달 대비, YoY: 12개월 전(작년 같은 달) 대비
    monthly["sales_mom"] = monthly["sales"].pct_change()
    monthly["sales_yoy"] = monthly["sales"].pct_change(periods=12)
    return monthly


def main() -> None:
    df = load_data(SUPERSTORE)

    print("=== 전체 요약 ===")
    for name, value in summarize(df).items():
        print(f"{name:>15}: {value:,.0f}")

    print("\n=== 카테고리별 매출 ===")
    print(sales_by_category(df).map("{:,.0f}".format).to_string())

    print("\n=== 월별 추이 (최근 12개월) ===")
    monthly = monthly_kpis(df)
    print(
        monthly.tail(12).to_string(
            formatters={
                "sales": "{:,.0f}".format,
                "profit": "{:,.0f}".format,
                "sales_mom": "{:+.1%}".format,
                "sales_yoy": "{:+.1%}".format,
            }
        )
    )


if __name__ == "__main__":
    main()
