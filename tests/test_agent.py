"""API를 호출하지 않고, 미리 정해 둔 응답을 돌려주는 가짜 클라이언트로 루프를 검사한다."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

import agent
from agent import run_agent

FINAL = {"answer": "답변", "suggested_actions": [{"action": "A", "reason": "R", "priority": "high"}]}


def function_call(name: str, arguments: dict, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(type="function_call", name=name, arguments=json.dumps(arguments), call_id=call_id)


def response(output: list, text: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        output=output,
        output_text=text,
        status="completed",
        incomplete_details=None,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def final_response() -> SimpleNamespace:
    return response([SimpleNamespace(type="message")], json.dumps(FINAL))


class FakeClient:
    def __init__(self, responses: list):
        self.queue = list(responses)
        self.requests = []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        # history는 이후에도 바뀌므로 호출 시점의 상태를 복사해 둔다.
        self.requests.append({**kwargs, "input": list(kwargs["input"])})
        return self.queue.pop(0)


@pytest.fixture
def df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-10", "2024-02-05"]),
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "C2"],
            "category": ["A", "B"],
            "sales": [100, 300],
            "profit": [10, 30],
        }
    )
    return df


def test_answers_directly_without_tools(df):
    client = FakeClient([final_response()])

    assert run_agent(client, "m", df, "sys", [{"role": "user", "content": "안녕"}]) == FINAL
    assert len(client.requests) == 1


def test_runs_requested_tools_and_returns_results(df):
    client = FakeClient(
        [
            # 한 번에 도구 두 개를 요청하는 경우
            response(
                [
                    function_call("get_summary", {"start_month": None, "end_month": None}, "call_1"),
                    function_call("category_breakdown", {"category": None, "start_month": None, "end_month": None}, "call_2"),
                ]
            ),
            final_response(),
        ]
    )
    history = [{"role": "user", "content": "매출 알려줘"}]

    assert run_agent(client, "m", df, "sys", history) == FINAL

    # 두 번째 요청에는 모델의 도구 요청과, call_id로 짝지은 실행 결과가 들어 있어야 한다.
    second_input = client.requests[1]["input"]
    outputs = {item["call_id"]: json.loads(item["output"]) for item in second_input if isinstance(item, dict) and item.get("type") == "function_call_output"}
    assert outputs["call_1"]["total_sales"] == 400
    assert [r["name"] for r in outputs["call_2"]["rows"]] == ["B", "A"]


def test_tool_error_is_sent_back_to_model(df):
    client = FakeClient([response([function_call("no_such_tool", {}, "call_1")]), final_response()])

    run_agent(client, "m", df, "sys", [{"role": "user", "content": "q"}])

    last = client.requests[1]["input"][-1]
    assert "error" in json.loads(last["output"])


def test_forces_answer_on_last_step(df, monkeypatch):
    monkeypatch.setattr(agent, "MAX_STEPS", 2)
    client = FakeClient([response([function_call("get_summary", {"start_month": None, "end_month": None}, "c1")]), final_response()])

    run_agent(client, "m", df, "sys", [{"role": "user", "content": "q"}])

    assert [r["tool_choice"] for r in client.requests] == ["auto", "none"]


def test_raises_when_model_never_stops_calling_tools(df, monkeypatch):
    monkeypatch.setattr(agent, "MAX_STEPS", 2)
    call = function_call("get_summary", {"start_month": None, "end_month": None}, "c")
    client = FakeClient([response([call]), response([call])])

    with pytest.raises(RuntimeError):
        run_agent(client, "m", df, "sys", [{"role": "user", "content": "q"}])
