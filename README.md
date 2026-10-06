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
| `monthly_trend` | `metric`, `last_n_months` | 지표 하나의 월별 값 + MoM/YoY |
| `category_breakdown` | `category`, `start_month`, `end_month` | 카테고리별 → (카테고리 지정 시) 하위 카테고리별 |
| `discount_impact` | `category` | 할인율 구간별 이익률 |

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
