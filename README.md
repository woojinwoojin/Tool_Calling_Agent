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
| 도구 | 인자 | 용도 |
|---|---|---|
| `get_summary` | `start_month`, `end_month` | 기간 전체 매출·이익·이익률·주문·고객·객단가 |
| `monthly_trend` | `metric`, `last_n_months`, `category` | 지표 하나의 월별 값 + MoM/YoY (전체 또는 카테고리) |
| `category_breakdown` | `category`, `start_month`, `end_month` | 카테고리별 → (카테고리 지정 시) 하위 카테고리별 + 합계 |
| `compare_periods` | 기준/비교 기간, `category` | 두 기간의 항목별 증감액·증감률·전체 변화 중 비중 |
| `discount_impact` | `category`, `sub_category` | 할인율 구간별 이익률 |

### 애매한 질문 실험에서 고친 것
| 질문 | 문제 | 해결 |
|---|---|---|
| "Furniture 이익률이 왜 낮아?" | 카테고리 합계를 LLM이 직접 계산 | `category_breakdown`에 `total` 추가 |
| "office supplies 쪽은 어때?" | 전체 매장 월별 추이를 카테고리 추이처럼 설명 | `monthly_trend`에 `category` 추가 |
| "지난달 매출이 왜 떨어졌어?" | 도구 9번 호출(입력 11K 토큰), 증감액을 LLM이 계산 | `compare_periods` 추가 → 5번 호출(입력 7.5K 토큰) |
| "Chairs 할인 영향 알려줘" | 하위 카테고리 미지원 → Furniture로 우회 | `discount_impact`에 `sub_category` 추가, 오류에 `hint` 추가 |
| "지역별 매출 1위는?" | "도구를 추가하라"를 제안 액션으로 냄 | 프롬프트: 액션은 비즈니스 액션만, 근거 없으면 빈 배열 |

이후 후보: `customer_analysis`, `detect_anomaly`

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
