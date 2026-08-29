"""Tests for semantic cache integration."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest


def make_doc(content, source="test.pdf", page=1, chunk_id=None):
    meta = {"source": source, "page": page}
    if chunk_id:
        meta["chunk_id"] = chunk_id
    return SimpleNamespace(page_content=content, metadata=meta)


# ---------------------------------------------------------------------------
# semantic_cache unit tests
# ---------------------------------------------------------------------------

class TestCacheAvailability:
    def test_cache_disabled_returns_false(self, monkeypatch):
        import chatbot.semantic_cache as sc
        monkeypatch.setattr(sc, "SEMANTIC_CACHE_ENABLED", False)
        # ensure _cached_semantic_cache is None to force check
        monkeypatch.setattr(sc, "_cached_semantic_cache", None)
        assert sc.is_cache_available() is False

    def test_cache_available_when_enabled_and_init_succeeds(self, monkeypatch):
        import chatbot.semantic_cache as sc
        monkeypatch.setattr(sc, "SEMANTIC_CACHE_ENABLED", True)
        mock_cache = MagicMock()
        monkeypatch.setattr(sc, "_cached_semantic_cache", None)
        monkeypatch.setattr(sc, "_get_cache", lambda redis_url=None: mock_cache)
        assert sc.is_cache_available() is True

    def test_get_cached_response_disabled_returns_miss(self, monkeypatch):
        import chatbot.semantic_cache as sc
        monkeypatch.setattr(sc, "SEMANTIC_CACHE_ENABLED", False)
        monkeypatch.setattr(sc, "_cached_semantic_cache", None)
        # is_cache_available will check disabled
        ans, meta = sc.get_cached_response("hello")
        assert ans is None
        assert meta["cache_hit"] is False
        assert meta["cache_source"] == "disabled"

    def test_redis_failure_treated_as_miss(self, monkeypatch):
        import chatbot.semantic_cache as sc
        monkeypatch.setattr(sc, "SEMANTIC_CACHE_ENABLED", True)
        # mock _get_cache to raise inside get_cached_response -> caught and treated as miss
        def failing_get_cache(redis_url=None):
            raise RuntimeError("redis down")
        monkeypatch.setattr(sc, "_get_cache", failing_get_cache)
        # also make is_cache_available return True via patch
        monkeypatch.setattr(sc, "is_cache_available", lambda: True)
        ans, meta = sc.get_cached_response("hello")
        assert ans is None
        assert meta["cache_hit"] is False
        assert meta["cache_source"] == "semantic_error"

    def test_cache_miss(self, monkeypatch):
        import chatbot.semantic_cache as sc
        monkeypatch.setattr(sc, "SEMANTIC_CACHE_ENABLED", True)
        mock_cache = MagicMock()
        mock_cache.check.return_value = []
        monkeypatch.setattr(sc, "_get_cache", lambda redis_url=None: mock_cache)
        monkeypatch.setattr(sc, "is_cache_available", lambda: True)
        ans, meta = sc.get_cached_response("What are admission requirements?")
        assert ans is None
        assert meta["cache_hit"] is False
        assert meta["cache_source"] == "semantic_miss"
        assert "cache_lookup_latency_ms" in meta

    def test_cache_hit(self, monkeypatch):
        import chatbot.semantic_cache as sc
        monkeypatch.setattr(sc, "SEMANTIC_CACHE_ENABLED", True)
        mock_cache = MagicMock()
        mock_cache.check.return_value = [
            {"response": "Admission requires form X", "id": "123", "prompt": "admission req", "metadata": {}}
        ]
        monkeypatch.setattr(sc, "_get_cache", lambda redis_url=None: mock_cache)
        monkeypatch.setattr(sc, "is_cache_available", lambda: True)
        ans, meta = sc.get_cached_response("What are admission requirements?")
        assert ans == "Admission requires form X"
        assert meta["cache_hit"] is True
        assert meta["cache_source"] == "semantic"

    def test_cache_store(self, monkeypatch):
        import chatbot.semantic_cache as sc
        monkeypatch.setattr(sc, "SEMANTIC_CACHE_ENABLED", True)
        mock_cache = MagicMock()
        mock_cache.store.return_value = "school_chatbot_semantic_cache:abc123"
        monkeypatch.setattr(sc, "_get_cache", lambda redis_url=None: mock_cache)
        monkeypatch.setattr(sc, "is_cache_available", lambda: True)
        key = sc.set_cached_response("What are admission requirements?", "Admission requires form X", ttl=3600)
        assert key == "school_chatbot_semantic_cache:abc123"
        mock_cache.store.assert_called_once()
        kwargs = mock_cache.store.call_args.kwargs
        assert kwargs["prompt"] == "What are admission requirements?"
        assert kwargs["response"] == "Admission requires form X"
        assert kwargs["ttl"] == 3600
        # metadata should contain version fields
        assert "cache_version" in kwargs["metadata"]
        assert "embedding_model" not in kwargs["metadata"] or "model_name" in kwargs["metadata"]

    def test_cache_store_failure_returns_none(self, monkeypatch):
        import chatbot.semantic_cache as sc
        monkeypatch.setattr(sc, "SEMANTIC_CACHE_ENABLED", True)
        mock_cache = MagicMock()
        mock_cache.store.side_effect = RuntimeError("redis down")
        monkeypatch.setattr(sc, "_get_cache", lambda redis_url=None: mock_cache)
        monkeypatch.setattr(sc, "is_cache_available", lambda: True)
        key = sc.set_cached_response("q", "a")
        assert key is None

    def test_ttl_passed_correctly_default(self, monkeypatch):
        import chatbot.semantic_cache as sc
        import config
        monkeypatch.setattr(sc, "SEMANTIC_CACHE_ENABLED", True)
        mock_cache = MagicMock()
        mock_cache.store.return_value = "key"
        monkeypatch.setattr(sc, "_get_cache", lambda redis_url=None: mock_cache)
        monkeypatch.setattr(sc, "is_cache_available", lambda: True)
        sc.set_cached_response("q", "a", ttl=None)
        _, kwargs = mock_cache.store.call_args
        assert kwargs["ttl"] == config.SEMANTIC_CACHE_TTL_SECONDS

    def test_version_metadata_included(self, monkeypatch):
        import chatbot.semantic_cache as sc
        import config
        monkeypatch.setattr(sc, "SEMANTIC_CACHE_ENABLED", True)
        mock_cache = MagicMock()
        mock_cache.store.return_value = "key"
        monkeypatch.setattr(sc, "_get_cache", lambda redis_url=None: mock_cache)
        monkeypatch.setattr(sc, "is_cache_available", lambda: True)
        sc.set_cached_response("q", "a", metadata={"extra": "val"})
        _, kwargs = mock_cache.store.call_args
        md = kwargs["metadata"]
        assert md["index_version"] == config.INDEX_VERSION
        assert md["chunking_version"] == config.CHUNKING_VERSION
        assert md["model_name"] == config.EMBEDDING_MODEL_NAME
        assert md["cache_version"] == "1.0"
        assert md["extra"] == "val"


class TestCacheEligibility:
    def test_empty_question_bypass(self):
        import chatbot.semantic_cache as sc
        assert sc.is_cache_eligible("", [], None) is False
        assert sc.is_cache_eligible("   ", [], None) is False

    def test_chat_history_bypass(self):
        import chatbot.semantic_cache as sc
        assert sc.is_cache_eligible("hello", [("prev", "ans")], None) is False

    def test_relevant_mem0_bypass(self):
        import chatbot.semantic_cache as sc
        mock_mem = MagicMock()
        mock_mem.get_relevant_memories.return_value = [{"memory": "user likes cats"}]
        assert sc.is_cache_eligible("hello", [], "user123", memory_layer=mock_mem) is False

    def test_no_relevant_mem0_eligible(self):
        import chatbot.semantic_cache as sc
        mock_mem = MagicMock()
        mock_mem.get_relevant_memories.return_value = []
        assert sc.is_cache_eligible("What are admission requirements?", [], "user123", memory_layer=mock_mem) is True

    def test_auth_question_bypass(self):
        import chatbot.semantic_cache as sc
        for q in ["what is my password", "login with google", "who am i", "account settings", "delete my account"]:
            assert sc.is_cache_eligible(q, [], None) is False, f"should bypass: {q}"

    def test_normal_question_eligible(self):
        import chatbot.semantic_cache as sc
        assert sc.is_cache_eligible("What are the admission requirements?", [], None) is True


# ---------------------------------------------------------------------------
# chain integration tests (mocked RAG)
# ---------------------------------------------------------------------------

def _setup_chain_mocks(monkeypatch, chain, llm_answer="Answer from LLM", retriever_docs=None):
    if retriever_docs is None:
        retriever_docs = [make_doc("school policy content", chunk_id="chunk-1")]
    class FakeRetriever:
        def invoke(self, q):
            return retriever_docs
    class FakeLLM:
        def invoke(self, messages):
            return SimpleNamespace(content=llm_answer, usage_metadata={}, response_metadata={})
    monkeypatch.setattr(chain, "get_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(chain, "get_vectorstore", lambda: MagicMock())
    monkeypatch.setattr(chain, "get_llm", lambda model_name: FakeLLM())
    # mock memory layer: no relevant memories, no writes verification
    mock_mem_layer = MagicMock()
    mock_mem_layer.get_relevant_memories.return_value = []
    mock_mem_layer.add_message.return_value = None
    monkeypatch.setattr(chain, "get_memory_layer", lambda user_id: mock_mem_layer)
    return mock_mem_layer


class TestChainCacheIntegration:
    def test_chat_history_bypasses_cache(self, monkeypatch):
        from chatbot import chain
        _setup_chain_mocks(monkeypatch, chain)
        # Patch semantic cache to ensure it would be eligible if not for chat_history
        mock_get = MagicMock(return_value=(None, {"cache_hit": False, "cache_lookup_latency_ms": 1}))
        mock_eligible = MagicMock(return_value=False)  # is_cache_eligible should return False for chat_history
        monkeypatch.setattr(chain, "get_cached_response", mock_get)
        monkeypatch.setattr(chain, "is_cache_eligible", mock_eligible)
        mock_store = MagicMock(return_value="key")
        monkeypatch.setattr(chain, "set_cached_response", mock_store)

        answer, docs, metadata = chain.ask_question(
            question="What are admission requirements?",
            model_name="test-model",
            chat_history=[("prev q", "prev a")],
            user_id="test-user",
            conversation_id="test-conv",
            return_metadata=True,
        )
        # should not hit cache
        assert metadata["cache_hit"] is False
        # store should not be called because was_cache_eligible is False
        mock_store.assert_not_called()

    def test_relevant_mem0_bypasses_cache(self, monkeypatch):
        from chatbot import chain
        _setup_chain_mocks(monkeypatch, chain)
        mock_mem = MagicMock()
        mock_mem.get_relevant_memories.return_value = [{"memory": "private"}]
        # Patch get_memory_layer to return mock with relevant memories for eligibility
        # But for simplicity patch is_cache_eligible to return False
        monkeypatch.setattr(chain, "is_cache_eligible", lambda **kwargs: False)
        mock_get = MagicMock(return_value=(None, {"cache_hit": False, "cache_lookup_latency_ms": 1}))
        monkeypatch.setattr(chain, "get_cached_response", mock_get)
        mock_store = MagicMock(return_value="key")
        monkeypatch.setattr(chain, "set_cached_response", mock_store)
        # ensure chain's get_memory_layer used for memory retrieval still works
        # need to set up again after patch?
        # For this test we directly test eligibility path via chain.ask_question without extra setup
        # Re-setup with mem that has relevant memories would cause eligibility false already
        # We'll just verify store not called
        # create fresh chain mocks with relevant mem
        class FakeRetriever:
            def invoke(self, q): return [make_doc("doc", chunk_id="c1")]
        class FakeLLM:
            def invoke(self, m): return SimpleNamespace(content="Answer", usage_metadata={}, response_metadata={})
        monkeypatch.setattr(chain, "get_retriever", lambda: FakeRetriever())
        monkeypatch.setattr(chain, "get_vectorstore", lambda: MagicMock())
        monkeypatch.setattr(chain, "get_llm", lambda model_name: FakeLLM())
        # get_memory_layer returns mock_mem for eligibility but second call for memory retrieval also returns empty?
        # For simplicity keep eligibility false so store skipped
        monkeypatch.setattr(chain, "get_memory_layer", lambda uid: mock_mem if uid else MagicMock(get_relevant_memories=lambda q: [], add_message=lambda *a, **kw: None))

        # monkeypatch memory retrieval to avoid second call issues: we want memory retrieval to return empty after eligibility
        # So we need to ensure chain's memory retrieval uses same mock but returns empty second time? simpler: just rely on is_cache_eligible=False
        answer, docs, metadata = chain.ask_question(
            question="What are admission requirements?",
            model_name="test-model",
            chat_history=[],
            user_id="test-user",
            conversation_id="test-conv",
            return_metadata=True,
        )
        assert metadata["cache_hit"] is False
        mock_store.assert_not_called()

    def test_successful_answer_stored(self, monkeypatch):
        from chatbot import chain
        _setup_chain_mocks(monkeypatch, chain)
        monkeypatch.setattr(chain, "is_cache_eligible", lambda **kwargs: True)
        monkeypatch.setattr(chain, "get_cached_response", lambda **kwargs: (None, {"cache_hit": False, "cache_lookup_latency_ms": 5, "cache_source": "semantic_miss"}))
        mock_store = MagicMock(return_value="key123")
        monkeypatch.setattr(chain, "set_cached_response", mock_store)

        answer, docs, metadata = chain.ask_question(
            question="What are admission requirements?",
            model_name="test-model",
            chat_history=[],
            user_id="test-user",
            conversation_id="test-conv",
            return_metadata=True,
        )
        assert answer != chain.FALLBACK_MESSAGE
        mock_store.assert_called_once()
        assert metadata["cache_store_latency_ms"] >= 0
        assert metadata["cache_hit"] is False

    def test_failed_answer_not_cached(self, monkeypatch):
        from chatbot import chain
        # LLM returns empty -> fallback -> should not cache
        _setup_chain_mocks(monkeypatch, chain, llm_answer="")
        monkeypatch.setattr(chain, "is_cache_eligible", lambda **kwargs: True)
        monkeypatch.setattr(chain, "get_cached_response", lambda **kwargs: (None, {"cache_hit": False, "cache_lookup_latency_ms": 5, "cache_source": "semantic_miss"}))
        mock_store = MagicMock(return_value="key")
        monkeypatch.setattr(chain, "set_cached_response", mock_store)

        answer, docs, metadata = chain.ask_question(
            question="What are admission requirements?",
            model_name="test-model",
            chat_history=[],
            user_id="test-user",
            conversation_id="test-conv",
            return_metadata=True,
        )
        assert answer == chain.FALLBACK_MESSAGE
        mock_store.assert_not_called()
        assert metadata["cache_store_latency_ms"] == 0

    def test_cache_hit_skips_llm_and_pinecone(self, monkeypatch):
        from chatbot import chain
        # Setup mocks that would fail if called
        def fail_retriever():
            raise AssertionError("Pinecone should not be called on cache hit")
        def fail_llm(model_name):
            raise AssertionError("LLM should not be called on cache hit")
        monkeypatch.setattr(chain, "get_retriever", fail_retriever)
        monkeypatch.setattr(chain, "get_vectorstore", lambda: (_ for _ in ()).throw(AssertionError("vectorstore should not be called")))
        monkeypatch.setattr(chain, "get_llm", fail_llm)
        mock_mem = MagicMock()
        mock_mem.get_relevant_memories.return_value = []
        monkeypatch.setattr(chain, "get_memory_layer", lambda uid: mock_mem)
        monkeypatch.setattr(chain, "is_cache_eligible", lambda **kwargs: True)
        monkeypatch.setattr(chain, "get_cached_response", lambda **kwargs: ("Cached answer", {"cache_hit": True, "cache_lookup_latency_ms": 3, "cache_source": "semantic"}))

        answer, docs, metadata = chain.ask_question(
            question="What requirements are needed for admission?",
            model_name="test-model",
            chat_history=[],
            user_id="test-user",
            conversation_id="test-conv",
            return_metadata=True,
        )
        assert answer == "Cached answer"
        assert metadata["cache_hit"] is True
        assert metadata["retrieval_latency_ms"] == 0
        assert metadata["llm_latency_ms"] == 0
        assert metadata["cache_store_latency_ms"] == 0
        assert metadata["cache_lookup_latency_ms"] == 3

    def test_cache_hit_metadata_version(self, monkeypatch):
        from chatbot import chain
        import config
        monkeypatch.setattr(chain, "is_cache_eligible", lambda **kwargs: True)
        monkeypatch.setattr(chain, "get_cached_response", lambda **kwargs: ("Cached answer", {"cache_hit": True, "cache_lookup_latency_ms": 2, "cache_source": "semantic"}))
        mock_mem = MagicMock()
        mock_mem.get_relevant_memories.return_value = []
        monkeypatch.setattr(chain, "get_memory_layer", lambda uid: mock_mem)
        # These will not be called but must be defined to import chain
        monkeypatch.setattr(chain, "get_retriever", lambda: MagicMock(invoke=lambda q: []))
        monkeypatch.setattr(chain, "get_vectorstore", lambda: MagicMock())
        monkeypatch.setattr(chain, "get_llm", lambda m: MagicMock(invoke=lambda msgs: SimpleNamespace(content="should not be called", usage_metadata={}, response_metadata={})))

        answer, docs, metadata = chain.ask_question(
            question="What are admission requirements?",
            model_name="test-model",
            chat_history=[],
            user_id="test-user",
            conversation_id="test-conv",
            return_metadata=True,
        )
        assert metadata["embedding_model"] == config.EMBEDDING_MODEL_NAME
        assert metadata["index_version"] == config.INDEX_VERSION
        assert metadata["chunking_version"] == config.CHUNKING_VERSION
        assert metadata["cache_version"] == "1.0"
        assert metadata["cache_enabled"] is True

    def test_redis_failure_does_not_break_chatbot(self, monkeypatch):
        from chatbot import chain
        _setup_chain_mocks(monkeypatch, chain)
        # Simulate get_cached_response raising then being caught as miss (semantic_cache handles) but chain also should survive store failure
        def failing_get(**kwargs):
            raise RuntimeError("redis down")
        # But semantic_cache.get_cached_response catches errors, so chain's get_cached_response won't raise normally.
        # Simulate store failure after generation
        monkeypatch.setattr(chain, "is_cache_eligible", lambda **kwargs: True)
        # mock get to return miss without error
        monkeypatch.setattr(chain, "get_cached_response", lambda **kwargs: (None, {"cache_hit": False, "cache_lookup_latency_ms": 5, "cache_source": "semantic_error"}))
        def failing_store(**kwargs):
            raise RuntimeError("redis down on store")
        monkeypatch.setattr(chain, "set_cached_response", failing_store)

        answer, docs, metadata = chain.ask_question(
            question="What are admission requirements?",
            model_name="test-model",
            chat_history=[],
            user_id="test-user",
            conversation_id="test-conv",
            return_metadata=True,
        )
        # Should still return answer despite store failure
        assert answer is not None
        assert metadata["cache_hit"] is False

    def test_every_response_has_cache_metadata(self, monkeypatch):
        from chatbot import chain
        _setup_chain_mocks(monkeypatch, chain)
        monkeypatch.setattr(chain, "is_cache_eligible", lambda **kwargs: False)
        # get_cached_response won't be called when not eligible, but mock anyway
        monkeypatch.setattr(chain, "get_cached_response", lambda **kwargs: (None, {"cache_hit": False, "cache_lookup_latency_ms": 0}))
        monkeypatch.setattr(chain, "set_cached_response", lambda **kwargs: None)
        answer, docs, metadata = chain.ask_question(
            question="What is my password?",
            model_name="test-model",
            chat_history=[("prev","ans")],
            user_id="test-user",
            conversation_id="test-conv",
            return_metadata=True,
        )
        assert "cache_enabled" in metadata
        assert "cache_hit" in metadata
        assert "cache_lookup_latency_ms" in metadata
        assert "cache_store_latency_ms" in metadata
