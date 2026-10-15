from studyguide.llm import CachedClient, FakeClient


def test_fake_client_records_prompts_and_scripts_responses():
    fake = FakeClient(["one", "two"])
    assert fake.generate("a") == "one"
    assert fake.generate("b") == "two"
    assert fake.prompts == ["a", "b"]


def test_fake_client_numbers_responses_by_default():
    assert FakeClient().generate("a") == "[fake response 1]"


def test_cached_client_only_calls_model_once_per_prompt(tmp_path):
    fake = FakeClient()
    client = CachedClient(fake, cache_dir=tmp_path, namespace="m")
    assert client.generate("same") == client.generate("same")
    assert len(fake.prompts) == 1
    client.generate("different")
    assert len(fake.prompts) == 2


def test_cache_is_separated_by_namespace(tmp_path):
    fake = FakeClient()
    CachedClient(fake, cache_dir=tmp_path, namespace="model-a").generate("p")
    CachedClient(fake, cache_dir=tmp_path, namespace="model-b").generate("p")
    assert len(fake.prompts) == 2
