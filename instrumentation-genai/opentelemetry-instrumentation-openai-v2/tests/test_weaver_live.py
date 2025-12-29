from time import sleep

def test_chat_completion_with_content(
    openai_client, vcr, weaver_container_v1_36, instrument_with_content_weaver_v1_36
):
    llm_model_value = "gpt-4o-mini"
    messages_value = [{"role": "user", "content": "Say this is a test"}]

    # vcr messes up with docker client, so we limit its scope to only the openai client calls
    with vcr.use_cassette('test_chat_completion_with_content.yaml'):
        response = openai_client.chat.completions.create(
            messages=messages_value,
            model=llm_model_value,
            stream=False,
        )

    sleep(7)  # wait for metrics to be exported

    full_report = weaver_container_v1_36.end_live_check()

    seen_metrics = full_report["statistics"]["seen_registry_metrics"]
    assert seen_metrics.get("gen_ai.client.operation.duration") == 1
    assert seen_metrics.get("gen_ai.client.token.usage") == 1

    seen_logs = full_report["statistics"]["seen_registry_events"]
    assert seen_logs.get("gen_ai.user.message") == 1
    assert seen_logs.get("gen_ai.choice") == 1

    # spans are special - they don't have identifiers and are not matched to semconv definition
    spans = [s for s in full_report["samples"] if s.get("span") is not None]
    assert len(spans) == 1

