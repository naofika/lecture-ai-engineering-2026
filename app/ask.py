"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask --chat
    uv run python -m app.ask --chat "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールを使わない「素の」呼び出し。通常モードは1回だけ質問し、--chat では会話履歴を保って続けて質問する。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    started_at = time.perf_counter()
    response = client.models.generate_content(model=MODEL, contents=question)
    elapsed_seconds = time.perf_counter() - started_at

    return format_response(response, elapsed_seconds)


def ask_with_history(chat, question: str) -> str:
    started_at = time.perf_counter()
    response = chat.send_message(question)
    elapsed_seconds = time.perf_counter() - started_at
    return format_response(response, elapsed_seconds)


def format_response(response, elapsed_seconds: float) -> str:
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage and usage.prompt_token_count is not None else "不明"
    output_tokens = usage.candidates_token_count if usage and usage.candidates_token_count is not None else "不明"
    print(f"[トークン数] 入力: {input_tokens}, 出力: {output_tokens}")
    print(f"[応答時間] {elapsed_seconds:.3f}秒")

    return response.text


def run_chat(chat, first_question: str | None = None) -> None:
    if first_question:
        print(f"[質問] {first_question}\n")
        print(ask_with_history(chat, first_question))

    print("質問を入力してください（終了: /exit または /quit）")
    while True:
        try:
            question = input("質問> ").strip()
        except EOFError:
            print()
            break
        if question.lower() in {"/exit", "/quit"}:
            break
        if not question:
            continue

        print(f"[質問] {question}\n")
        print(ask_with_history(chat, question))


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--chat", action="store_true", help="会話履歴を保ったまま続けて質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if args.chat:
        try:
            client = genai.Client(api_key=get_api_key())
            chat = client.chats.create(model=MODEL)
            run_chat(chat, args.question)
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            raise SystemExit(1)
        return

    question = args.question or DEFAULT_QUESTION
    print(f"[質問] {question}\n")
    try:
        print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
