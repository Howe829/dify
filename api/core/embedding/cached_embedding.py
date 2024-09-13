import base64
import logging
from typing import Optional, cast

import numpy as np
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select, insert

from core.model_manager import ModelInstance
from core.model_runtime.entities.model_entities import ModelPropertyKey
from core.model_runtime.model_providers.__base.text_embedding_model import TextEmbeddingModel
from core.rag.datasource.entity.embedding import Embeddings
from extensions.ext_database import transactional, database
from extensions.ext_redis import redis_client
from libs import helper
from models.dataset import Embedding

logger = logging.getLogger(__name__)


class CacheEmbedding(Embeddings):
    def __init__(self, model_instance: ModelInstance, user: Optional[str] = None) -> None:
        self._model_instance = model_instance
        self._user = user

    async def _get_embedding_from_db(self, texts: list[str]) -> (list[list[float]], list[int]):
        text_embeddings = [None for _ in range(len(texts))]
        embedding_queue_indices = []
        for i, text in enumerate(texts):
            hash = helper.generate_text_hash(text)
            query = select(Embedding).filter_by(model_name=self._model_instance.model,
                                                hash=hash,
                                                provider_name=self._model_instance.provider
                                                )
            embedding = await database.fetch_one(query)
            if embedding:
                text_embeddings[i] = embedding.get_embedding()
            else:
                embedding_queue_indices.append(i)
        return text_embeddings, embedding_queue_indices

    @transactional
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs in batches of 10."""
        # use doc embedding cache or store if not exists
        text_embeddings, embedding_queue_indices = await self._get_embedding_from_db(texts)
        if not embedding_queue_indices:
            return text_embeddings
        embedding_queue_texts = [texts[i] for i in embedding_queue_indices]
        embedding_queue_embeddings = []
        try:
            model_type_instance = cast(TextEmbeddingModel, self._model_instance.model_type_instance)
            model_schema = model_type_instance.get_model_schema(self._model_instance.model,
                                                                self._model_instance.credentials)
            max_chunks = model_schema.model_properties[ModelPropertyKey.MAX_CHUNKS] \
                if model_schema and ModelPropertyKey.MAX_CHUNKS in model_schema.model_properties else 1
            for i in range(0, len(embedding_queue_texts), max_chunks):
                batch_texts = embedding_queue_texts[i:i + max_chunks]

                embedding_result = self._model_instance.invoke_text_embedding(
                    texts=batch_texts,
                    user=self._user
                )

                for vector in embedding_result.embeddings:
                    try:
                        normalized_embedding = (vector / np.linalg.norm(vector)).tolist()
                        embedding_queue_embeddings.append(normalized_embedding)
                    except Exception as e:
                        logging.exception('Failed transform embedding: ', e)
            cache_embeddings = []
            try:
                embedding_caches = []
                for i, embedding in zip(embedding_queue_indices, embedding_queue_embeddings):
                    text_embeddings[i] = embedding
                    hash = helper.generate_text_hash(texts[i])
                    if hash not in cache_embeddings:
                        embedding_cache = Embedding(model_name=self._model_instance.model,
                                                    hash=hash,
                                                    provider_name=self._model_instance.provider)
                        embedding_cache.set_embedding(embedding)

                        embedding_caches.append(embedding_cache.dict())
                        cache_embeddings.append(hash)

                stmt = insert(Embedding).values(embedding_caches)
                await database.execute(stmt)
            except IntegrityError:
                pass
        except Exception as ex:
            logger.error('Failed to embed documents: %s', ex)
            raise ex

        return text_embeddings

    def _get_embedding_cache_key(self, text: str) -> str:
        hash_str = helper.generate_text_hash(text)
        return f'{self._model_instance.provider}_{self._model_instance.model}_{hash_str}'

    async def _get_embedding_from_cache(self, text: str) -> list[float] | None:
        embedding_cache_key = self._get_embedding_cache_key(text)
        embedding = await redis_client.get(embedding_cache_key)
        if not embedding:
            return None
        await redis_client.expire(embedding_cache_key, 600)
        return list(np.frombuffer(base64.b64decode(embedding), dtype="float"))

    async def _save_embedding_to_cache(self, text: str, embedding_results: list[float]) -> None:
        embedding_cache_key = self._get_embedding_cache_key(text)
        try:
            # encode embedding to base64
            embedding_vector = np.array(embedding_results)
            vector_bytes = embedding_vector.tobytes()
            # Transform to Base64
            encoded_vector = base64.b64encode(vector_bytes)
            # Transform to string
            encoded_str = encoded_vector.decode("utf-8")
            await redis_client.setex(embedding_cache_key, 600, encoded_str)

        except IntegrityError:
            database.session.rollback()
        except:
            logging.exception('Failed to add embedding to redis')

    async def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        # use doc embedding cache or store if not exists
        embedding = await self._get_embedding_from_cache(text)
        if embedding:
            return embedding

        try:
            embedding_result = self._model_instance.invoke_text_embedding(
                texts=[text],
                user=self._user
            )

            embedding_results = embedding_result.embeddings[0]
            embedding_results = (embedding_results / np.linalg.norm(embedding_results)).tolist()
        except Exception as ex:
            raise ex
        await self._save_embedding_to_cache(text, embedding_results)

        return embedding_results
