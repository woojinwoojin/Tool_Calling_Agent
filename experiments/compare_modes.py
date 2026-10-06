"""세 가지 답변 방식의 속도·비용·답변을 같은 질문으로 비교하는 실험.

A. parallel: 관점이 다른 fast 3개를 동시에 실행 → 종합 LLM 호출 1번
B. wide:     LLM 왕복 2번, 한 라운드에 도구 최대 6개
C. careful:  결과를 보고 단계적으로 좁혀 가는 기존 신중 모드

실행: .\\.venv\\Scripts\\python experiments\\compare_modes.py
결과: experiments/results.md (표 + 답변 원문)
"""

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

from agent import ANSWER_FORMAT, DEFAULT_MODEL, MODES, Mode, build_system_prompt, run_agent  # noqa: E402
from analysis import SUPERSTORE, load_data  # noqa: E402

QUESTIONS = [
    "Furniture 이익률이 왜 낮아?",
    "지난달 매출이 왜 떨어졌어?",
    "Central 지역 이익률이 왜 낮아?",
]
REPEATS = 2  # 시간은 편차가 커서 여러 번 돌려 평균을 낸다. 답변 원문은 첫 번째 것만 저장한다.

# A: 같은 fast를 여러 번 돌리면 같은 도구를 고르므로, 실행마다 관점을 나눠 준다.
ANGLES = [
    "이번에는 상품 부문(category, sub_category) 관점에서만 분석하세요.",
    "이번에는 할인(discount_impact) 관점에서만 분석하세요.",
    "이번에는 지역·고객 세그먼트·기간 관점에서만 분석하세요.",
]
SYNTHESIS_PROMPT = """당신은 시니어 데이터 분석가입니다. 같은 질문에 대해 서로 다른 관점의 분석가 3명이 쓴 답변을 하나로 종합하세요.
- 숫자는 아래 답변에 있는 것만 인용하고, 새로 계산하지 마세요.
- 답변끼리 겹치는 내용은 한 번만 쓰고, 서로 다른 원인은 함께 정리하세요.
- answer는 5~8문장, suggested_actions는 데이터 근거가 있는 비즈니스 액션만 3개 이내로 쓰세요."""

# agent.py의 standard 모드가 이 설정에서 나왔다. 측정 당시 조건을 재현하려고 여기 그대로 둔다.
WIDE = Mode(
    name="wide",
    max_steps=2,
    max_tool_calls=6,
    reasoning_effort="low",
    guide="""[넓은 빠른 답변 모드]
- 원인이 될 만한 여러 관점(부문, 할인, 지역, 기간)을 고려해, 필요한 도구를 최대 6개까지 한 번에 동시에 호출하세요.
  결과를 받은 뒤에는 도구를 더 호출할 수 없습니다.
- breakdown과 compare_periods 결과에는 합계(total)가 들어 있으므로 get_summary를 따로 부를 필요가 없습니다.
- answer는 5~8문장으로, 결론과 핵심 숫자만 쓰세요.
- suggested_actions는 3개 이내로 제안하세요.""",
)


class CountingClient:
    """OpenAI 클라이언트를 감싸 토큰 사용량을 센다. 여러 스레드에서 함께 써도 되도록 잠근다."""

    def __init__(self, client: OpenAI):
        self.client = client
        self.input_tokens = self.output_tokens = self.llm_calls = 0
        self.lock = threading.Lock()
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        response = self.client.responses.create(**kwargs)
        with self.lock:
            self.input_tokens += response.usage.input_tokens
            self.output_tokens += response.usage.output_tokens
            self.llm_calls += 1
        return response


def count_tools(history: list) -> int:
    return sum(1 for item in history if isinstance(item, dict) and item.get("type") == "function_call_output")


def run_single(client, model, df, instructions, question, mode) -> tuple[dict, int]:
    history = [{"role": "user", "content": question}]
    answer = run_agent(client, model, df, instructions, history, mode)
    return answer, count_tools(history)


def run_parallel(client, model, df, instructions, question) -> tuple[dict, int]:
    fast = MODES["fast"]
    modes = [replace(fast, guide=f"{fast.guide}\n- {angle}") for angle in ANGLES]
    with ThreadPoolExecutor(max_workers=len(modes)) as pool:
        results = list(pool.map(lambda m: run_single(client, model, df, instructions, question, m), modes))

    answers = [answer for answer, _ in results]
    response = client.responses.create(
        model=model,
        instructions=SYNTHESIS_PROMPT,
        input=f"질문: {question}\n\n" + json.dumps(answers, ensure_ascii=False, indent=2),
        text={"format": ANSWER_FORMAT},
        reasoning={"effort": "low"},
    )
    return json.loads(response.output_text), sum(tools for _, tools in results)


def measure(method: str, client: OpenAI, model, df, instructions, question) -> dict:
    counting = CountingClient(client)
    started = time.perf_counter()
    if method == "A. parallel":
        answer, tools = run_parallel(counting, model, df, instructions, question)
    elif method == "B. wide":
        answer, tools = run_single(counting, model, df, instructions, question, WIDE)
    else:
        answer, tools = run_single(counting, model, df, instructions, question, MODES["careful"])
    return {
        "time": time.perf_counter() - started,
        "llm_calls": counting.llm_calls,
        "tools": tools,
        "input": counting.input_tokens,
        "output": counting.output_tokens,
        "answer": answer,
    }


def format_answer(answer: dict) -> str:
    lines = [answer["answer"], ""]
    for action in answer["suggested_actions"]:
        lines.append(f"- ({action['priority']}) {action['action']} — {action['reason']}")
    return "\n".join(lines)


def main() -> None:
    load_dotenv()
    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    df = load_data(SUPERSTORE)
    instructions = build_system_prompt(df, SUPERSTORE.currency)
    methods = ["A. parallel", "B. wide", "C. careful"]

    rows, answers = [], []
    for question in QUESTIONS:
        for method in methods:
            runs = [measure(method, client, model, df, instructions, question) for _ in range(REPEATS)]
            avg = {key: sum(r[key] for r in runs) / REPEATS for key in ("time", "llm_calls", "tools", "input", "output")}
            times = ", ".join(f"{r['time']:.1f}" for r in runs)
            row = (
                f"| {question} | {method} | {avg['time']:.1f}초 ({times}) | {avg['llm_calls']:.1f} | {avg['tools']:.1f} "
                f"| {avg['input']:,.0f} | {avg['output']:,.0f} |"
            )
            print(row, flush=True)
            rows.append(row)
            answers.append(f"### {question} — {method}\n\n{format_answer(runs[0]['answer'])}\n")

    header = [
        f"# 답변 방식 비교 ({model}, 질문당 {REPEATS}회 평균)",
        "",
        "| 질문 | 방식 | 시간 (각 회) | LLM 호출 | 도구 호출 | 입력 토큰 | 출력 토큰 |",
        "|---|---|---|---|---|---|---|",
    ]
    out = Path(__file__).parent / "results.md"
    out.write_text("\n".join(header + rows) + "\n\n## 답변 원문 (첫 번째 실행)\n\n" + "\n".join(answers), encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
