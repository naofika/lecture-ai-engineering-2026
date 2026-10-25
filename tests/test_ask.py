from types import SimpleNamespace

from app import ask as ask_module


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=7),
        text="回答",
    )
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: response))
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    elapsed_times = iter([10.0, 11.234])
    monkeypatch.setattr(ask_module.time, "perf_counter", lambda: next(elapsed_times))

    assert ask_module.ask("質問") == "回答"
    output = capsys.readouterr().out
    assert "[トークン数] 入力: 12, 出力: 7" in output
    assert "[応答時間] 1.234秒" in output


def test_chat_mode_keeps_history_between_questions(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=7),
        text="回答",
    )

    class Chat:
        def __init__(self):
            self.questions = []

        def send_message(self, question):
            self.questions.append(question)
            return response

    chat = Chat()
    client = SimpleNamespace(chats=SimpleNamespace(create=lambda **kwargs: chat))
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr("sys.argv", ["app.ask", "--chat", "最初の質問"])
    inputs = iter(["続きの質問", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    ask_module.main()

    assert chat.questions == ["最初の質問", "続きの質問"]
    assert capsys.readouterr().out.count("回答") == 2