from unittest.mock import MagicMock

import pytest
from api.core.embedding.cached_embedding import CacheEmbedding
from core.entities.provider_configuration import ProviderModelBundle, ProviderConfiguration
from core.entities.provider_entities import SystemConfiguration, CustomConfiguration, CustomProviderConfiguration
from core.model_manager import ModelInstance
from core.model_runtime.entities.model_entities import ModelType
from core.model_runtime.model_providers import ModelProviderFactory
from models.provider import ProviderType


@pytest.fixture
def mock_session():
    return MagicMock()


@pytest.fixture
def model_instance():
    credentials = {"base_url": "http://localhost:11434/", "context_size": 4096}

    provider_instance = ModelProviderFactory().get_provider_instance("ollama")
    model_type_instance = provider_instance.get_model_instance(ModelType.TEXT_EMBEDDING)
    provider_model_bundle = ProviderModelBundle(
        configuration=ProviderConfiguration(
            tenant_id="1",
            provider=provider_instance.get_provider_schema(),
            preferred_provider_type=ProviderType.CUSTOM,
            using_provider_type=ProviderType.CUSTOM,
            system_configuration=SystemConfiguration(enabled=False),
            custom_configuration=CustomConfiguration(provider=CustomProviderConfiguration(credentials=credentials)),
            model_settings=[],
        ),
        provider_instance=provider_instance,
        model_type_instance=model_type_instance,
    )
    model_instance = ModelInstance(provider_model_bundle=provider_model_bundle, model="nomic-embed-text")
    return model_instance


@pytest.mark.asyncio
async def test_embed_query(model_instance):
    embedding = CacheEmbedding(model_instance)
    embeddings = await embedding.embed_query("test docs")
    assert len(embeddings) > 1


@pytest.mark.asyncio
async def test_embed_documents(model_instance):
    embedding = CacheEmbedding(model_instance)
    embeddings = await embedding.embed_documents(["apple", "banana", "grapes"])
    print(embeddings)
    assert len(embeddings) > 1
