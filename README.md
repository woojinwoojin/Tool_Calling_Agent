일단# 2주차 — 분석 함수 분리 + Tool Calling (첫 Agent)

## 목표
분석 로직을 함수로 나누고, LLM이 사용자 질문에 맞춰 필요한 함수를 스스로 고르게 한다.

## 흐름
```
질문 → LLM이 tool 선택 → Python 분석 함수 실행 → 결과 → LLM → 설명 + Action 제안
```

## 설계 결정
- **LLM:** OpenAI Responses API(`gpt-5.4-mini`)로 루프를 먼저 완성 → 이후 Anthropic 추가하며 차이 비교
- **인터페이스:** CLI 대화 루프 (UI는 3주차)
- **시작점:** 1주차 `analysis.py`를 가져와 `sub_category`, `discount` 역할 추가
- **도구 결과:** 원본 행이 아니라 요약 숫자만 JSON으로 반환. 잘못된 인자는 예외 대신 `{"error": ...}`로 돌려줘 LLM이 스스로 고치게 함

## 도구 (`tools.py`)
인자를 두 종류로 나눈다.
- **`filters`** — 어떤 행만 볼지. 모든 도구 공통: `{"category", "sub_category", "region", "segment"}` (안 쓰는 항목은 null)
- **`group_by`** — 무엇으로 나눠 볼지. 나눠 보는 도구만: `category` / `sub_category` / `region` / `segment`

| 도구 | 인자 (`filters` 외) | 용도 |
|---|---|---|
| `get_summary` | 기간 | 매출·이익·이익률·주문·고객·객단가 |
| `monthly_trend` | `metric`, `last_n_months` | 지표 하나의 월별 값 + MoM/YoY |
| `breakdown` | `group_by`, 기간 | 기준별 매출·매출 비중·이익·이익률 + 합계 |
| `compare_periods` | 기준/비교 기간, `group_by` | 두 기간의 항목별 증감액·증감률·전체 변화 중 비중 |
| `discount_impact` | — | 할인율 구간별 이익률 |
| `customer_analysis` | 기간 | 신규/기존 고객, 재구매율, 상위 10% 고객 매출 비중 (신규 = 회사 전체에서 첫 구매) |
| `detect_anomaly` | `metric`(sales/orders/customers) | 계절성·연간 수준을 반영한 기대값 대비 크게 벗어난 달 |

`detect_anomaly`는 이익을 지원하지 않는다. 이익은 0 근처나 음수인 달이 있어 "기대값 대비 %"가 수천 %로 튄다.

### 애매한 질문 실험에서 고친 것
| 질문 | 문제 | 해결 |
|---|---|---|
| "Furniture 이익률이 왜 낮아?" | 카테고리 합계를 LLM이 직접 계산 | `category_breakdown`에 `total` 추가 |
| "office supplies 쪽은 어때?" | 전체 매장 월별 추이를 카테고리 추이처럼 설명 | `monthly_trend`에 `category` 추가 |
| "지난달 매출이 왜 떨어졌어?" | 도구 9번 호출(입력 11K 토큰), 증감액을 LLM이 계산 | `compare_periods` 추가 → 5번 호출(입력 7.5K 토큰) |
| "Chairs 할인 영향 알려줘" | 하위 카테고리 미지원 → Furniture로 우회 | `discount_impact`에 `sub_category` 추가, 오류에 `hint` 추가 |
| "지역별 매출 1위는?" | "도구를 추가하라"를 제안 액션으로 냄 | 프롬프트: 액션은 비즈니스 액션만, 근거 없으면 빈 배열 → 이후 `dimension_breakdown` 추가 |
| "Central 지역 이익률이 왜 낮아?" | 필터가 `category`뿐이라 9번 호출(11.5K), 할인 분석은 전체 지역 수치로 대체 | 공통 `filters` + `group_by` 분리 → 6번 호출(9.1K), Central 전용 할인 수치 사용 |
| "Furniture 이익률이 왜 낮아?" | 기준·비교 기간이 같은 `compare_periods` 호출 | 같은 기간이면 error + `breakdown` 안내 |

**`filters` 도입의 비용:** 모든 도구 스키마에 필터 4개가 들어가 요청마다 입력이 약 900~1,800 토큰 늘었다
(Chairs 질문 2.6K → 4.4K). 조합 질문은 호출 수가 줄어 이득이지만, 단순 질문은 더 비싸졌다.

## 실행 방법
```powershell
# 1. 데이터: 1주차와 같은 data/Sample - Superstore.csv (git에는 포함 안 함)
# 2. 환경 준비
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env   # 그다음 .env에 API 키 입력

# 3. 실행
.\.venv\Scripts\python agent.py   # CLI 대화 (도구 호출 과정은 [tool] 로그로 표시)

# 4. 테스트 (API 호출 없음 — 가짜 클라이언트로 루프 검사)
.\.venv\Scripts\python -m pytest
```

## 예시
"지난달 매출이 왜 떨어졌어?" → 월별 매출 → 전월 대비 성장률 → 카테고리별 비교 → 감소 카테고리 탐색

## 완료 체크
- [x] 분석 함수 분리
- [x] Tool 스키마 정의
- [x] Tool Calling 루프 직접 구현 (프레임워크 없이)
- [x] Structured Output으로 Suggested Action 생성

## 블로그 주제
Tool Calling으로 챗봇을 Agent처럼 만드는 과정
