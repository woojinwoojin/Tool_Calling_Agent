일단# 2주차 — 분석 함수 분리 + Tool Calling (첫 Agent)

## 목표
분석 로직을 함수로 나누고, LLM이 사용자 질문에 맞춰 필요한 함수를 스스로 고르게 한다.

## 흐름
```
질문 → LLM이 tool 선택 → Python 분석 함수 실행 → 결과 → LLM → 설명 + Action 제안
```

## 설계 결정
- **LLM:** OpenAI Responses API(`gpt-5.4-mini`)로 완성. Anthropic 지원은 보류 (REPORT.md 6장)
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

## 답변 모드 (`agent.py`의 `MODES`)
응답 시간은 LLM 왕복 횟수, 추론 토큰, 출력 길이가 좌우한다. 도구는 로컬에서 바로 실행되므로 호출 개수는 속도에 거의 영향이 없다.

| | `fast` | `standard` (기본) | `careful` |
|---|---|---|---|
| 용도 | 빠른 확인 | 대부분의 질문 | 결과를 보고 한 단계씩 파고들어야 할 때 |
| LLM 왕복 | 최대 2번 | 최대 2번 | 최대 6번 |
| 도구 호출 | 최대 3개 | 최대 6개 (한 라운드에 넓게) | 최대 10개 |
| 추론 강도 | `none` | `low` | `medium` |
| 답변 | 3~5문장, 액션 1개 | 5~8문장, 액션 3개 | 비교 기준을 밝히고 액션 3개 |
| 측정 시간 | 약 4초 | 7~9초 | 9~20초 |

상한을 넘은 도구 호출에도 error를 output으로 돌려준다. 모든 `function_call`에는 짝이 되는 output이 있어야 하기 때문이다.

### 실험: 병렬 fast + 종합 vs 넓은 fast vs careful
"fast가 2~3배 빠르니 관점이 다른 fast 3개를 병렬로 돌리고 종합하면 어떨까?"를 측정했다.
([`experiments/compare_modes.py`](experiments/compare_modes.py), 결과와 답변 원문은 [`experiments/results.md`](experiments/results.md))

| 질문 (2회 평균) | A. 병렬 fast 3 + 종합 | B. 넓은 fast (→ `standard`) | C. careful |
|---|---|---|---|
| Furniture 이익률이 왜 낮아? | 9.5초 / 입력 16.6K | **7.1초** / 5.6K | 8.8초 / 5.4K |
| 지난달 매출이 왜 떨어졌어? | 10.2초 / 18.3K | **8.9초** / 6.4K | 20.1초 / 20.1K |
| Central 지역 이익률이 왜 낮아? | 8.8초 / 16.6K | **8.5초** / 6.1K | 14.8초 / 13.9K |

- **A는 B보다 늘 느리고 토큰은 약 3배.** 병렬이어도 가장 느린 fast(4~5초)에 종합 호출(4~5초)이 더해진다.
- **A의 종합 단계가 근거 없는 인과를 만들었다.** "Tables 적자"와 "고할인 구간 적자"라는 서로 다른 관점의 결과를
  "Tables에 높은 할인이 붙어서"로 이어 붙였지만, Tables의 할인을 조회한 도구는 없었다. 핵심 숫자(감소분의 80.7%)가 빠지기도 했다.
- **B와 C는 답변 내용이 대부분 비슷했다.** 차이는 C만 결과를 보고 한 단계 더 들어간다는 점(Technology 감소 → Machines/Phones/Copiers).
- 결론: 같은 모델로 관점만 나눈 병렬화보다 **한 번의 호출에서 도구를 넓게 부르는 것**이 빠르고 정확하다.
  병렬이 의미 있으려면 각 작업이 여러 라운드를 거쳐 깊게 파고드는 경우여야 한다 (→ 프로젝트 4 Multi-Agent).
- 한계: 질문 3개 × 2회라 경향만 본 것. careful은 같은 질문에서도 15.3초 / 24.9초처럼 편차가 컸다.

## 실행 방법
```powershell
# 1. 데이터: 1주차와 같은 data/Sample - Superstore.csv (git에는 포함 안 함)
# 2. 환경 준비
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env   # 그다음 .env에 API 키 입력

# 3. 실행
.\.venv\Scripts\python agent.py                  # CLI 대화, 기본(standard) 모드 (도구 호출 과정은 [tool] 로그로 표시)
.\.venv\Scripts\python agent.py --mode careful   # fast / standard / careful. 대화 중에는 /fast, /standard, /careful 로 전환
.\.venv\Scripts\python experiments\compare_modes.py  # 답변 방식 비교 실험 (API 호출 약 80번)

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
Tool Calling으로 챗봇을 Agent처럼 만드는 과정 — 결과와 발견은 [REPORT.md](REPORT.md)
