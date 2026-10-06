"""질문 → LLM이 도구 선택 → Python 실행 → 결과 → LLM → 답변 + 제안 액션.

프레임워크 없이 OpenAI Responses API로 tool calling 루프를 직접 구현한다.
실행: .\\.venv\\Scripts\\python agent.py
"""

import json
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from analysis import SUPERSTORE, load_data
from tools import TOOLS, run_tool

DEFAULT_MODEL = "gpt-5.4-mini"
# 도구 호출이 끝없이 이어져 비용이 새지 않도록 LLM 호출 횟수에 상한을 둔다.
MAX_STEPS = 6

SYSTEM_PROMPT = """당신은 이커머스 회사의 시니어 데이터 분석가입니다.
사용자 질문에 답하기 위해 필요한 분석 도구를 골라 호출하고, 도구 결과만 근거로 한국어로 답하세요.

- 숫자는 도구 결과에서만 인용하세요. 필요한 숫자가 없으면 추측하지 말고 도구를 더 호출하세요.
- 원인을 묻는 질문은 전체 → 기간/카테고리 → 하위 카테고리/할인 순으로 좁혀 가며 확인하세요.
- 월별 변화는 MoM과 YoY를 함께 보세요. 매출은 계절성이 강하므로 MoM만으로 급락/급등이라고 단정하지 마세요.
- 데이터로 확인할 수 없는 원인은 가설로 표시하세요.
- 도구가 error를 돌려주면 available 값 등을 참고해 인자를 고쳐 다시 호출하세요.
- 금액에는 통화 단위를 붙이세요.
- suggested_actions는 데이터 근거가 있는 비즈니스 액션(가격, 할인, 상품 구성, 마케팅 등)만 우선순위 순으로 3개 이내로 제안하세요.
  "추가 분석", "데이터 확인", "도구 추가" 같은 분석 작업은 액션이 아닙니다. 도구로 확인할 수 없는 한계는 answer에 쓰세요.
  근거가 되는 도구 결과가 없으면 suggested_actions는 빈 배열로 두세요.

데이터 정보:
{data_info}"""

# 최종 답변 형식 (Structured Output). strict 모드 규칙은 도구 스키마와 같다.
ANSWER_FORMAT = {
    "type": "json_schema",
    "name": "analysis_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "질문에 대한 분석 답변 (근거 숫자 포함)"},
            "suggested_actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "reason": {"type": "string", "description": "이 액션을 제안하는 데이터 근거"},
                        "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    },
                    "required": ["action", "reason", "priority"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["answer", "suggested_actions"],
        "additionalProperties": False,
    },
}


def build_system_prompt(df: pd.DataFrame, currency: str) -> str:
    # LLM이 올바른 인자(월, 필터 값)를 고를 수 있도록 데이터의 범위만 알려준다.
    # 하위 카테고리는 개수가 많아서 넣지 않는다. 틀리면 도구의 error에 available 목록이 온다.
    data_info = {
        "period": f"{df['date'].min():%Y-%m} ~ {df['date'].max():%Y-%m}",
        "currency": currency,
        **{f"{col}_values": sorted(df[col].unique()) for col in ("category", "region", "segment") if col in df},
    }
    return SYSTEM_PROMPT.format(data_info=json.dumps(data_info, ensure_ascii=False))


def log(message: str) -> None:
    # 답변(stdout)과 섞이지 않도록 진행 상황은 stderr로 출력한다.
    print(message, file=sys.stderr)


def run_agent(client: OpenAI, model: str, df: pd.DataFrame, instructions: str, history: list) -> dict:
    """history(대화 기록)에 질문이 들어 있는 상태로 호출한다. 도구 호출과 답변이 history에 추가된다."""
    input_tokens = output_tokens = 0

    for step in range(1, MAX_STEPS + 1):
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=history,
            tools=TOOLS,
            # 마지막 차례에는 도구를 못 쓰게 해서 지금까지의 결과로 답하게 한다.
            tool_choice="none" if step == MAX_STEPS else "auto",
            text={"format": ANSWER_FORMAT},
            reasoning={"effort": "low"},
        )
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens

        # API는 이전 요청을 기억하지 않으므로, 모델의 출력(도구 요청 포함)도 대화 기록에 그대로 남긴다.
        history.extend(response.output)

        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            log(f"[usage] {model} steps={step} input={input_tokens} output={output_tokens}")
            if response.status != "completed":
                raise RuntimeError(f"응답이 완료되지 않았습니다: {response.incomplete_details}")
            return json.loads(response.output_text)

        for call in calls:
            log(f"[tool] {call.name}({call.arguments})")
            history.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,  # 어떤 요청에 대한 결과인지 짝을 맞춘다
                    "output": run_tool(df, call.name, call.arguments),
                }
            )

    raise RuntimeError(f"{MAX_STEPS}번 안에 답변을 받지 못했습니다.")


def print_answer(answer: dict) -> None:
    print("\n" + answer["answer"])
    if answer["suggested_actions"]:
        print("\n[제안 액션]")
        for i, action in enumerate(answer["suggested_actions"], 1):
            print(f"{i}. ({action['priority']}) {action['action']}")
            print(f"   근거: {action['reason']}")
    print()


def main() -> None:
    load_dotenv()
    client = OpenAI()  # .env의 OPENAI_API_KEY를 사용한다
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    df = load_data(SUPERSTORE)
    instructions = build_system_prompt(df, SUPERSTORE.currency)

    # 후속 질문("그럼 Tables는?")이 앞 대화를 참고할 수 있도록 기록을 이어서 쓴다.
    history: list = []
    print("질문을 입력하세요. (종료: 빈 줄 또는 Ctrl+C)")
    while True:
        try:
            question = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        checkpoint = len(history)
        history.append({"role": "user", "content": question})
        try:
            print_answer(run_agent(client, model, df, instructions, history))
        except Exception as e:  # API 키 누락, 크레딧 부족 등 — 대화는 계속한다
            log(f"[error] {e}")
            del history[checkpoint:]  # 실패한 질문의 기록은 지워서 다음 질문에 섞이지 않게 한다


if __name__ == "__main__":
    main()
