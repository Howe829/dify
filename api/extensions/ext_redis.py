import redis
import aioredis
from redis.connection import Connection, SSLConnection

# redis_client = redis.Redis()
redis_client = aioredis.Redis(
    host='localhost',
    port=6379,
    username="",
    password="difyai123456",
    db=1,
    encoding="utf-8",
    encoding_errors="strict",
    decode_responses=False,
)

def init_app(app):
    connection_class = Connection
    if app.config.get("REDIS_USE_SSL"):
        connection_class = SSLConnection

    redis_client.connection_pool = redis.ConnectionPool(
        **{
            "host": app.config.get("REDIS_HOST"),
            "port": app.config.get("REDIS_PORT"),
            "username": app.config.get("REDIS_USERNAME"),
            "password": app.config.get("REDIS_PASSWORD"),
            "db": app.config.get("REDIS_DB"),
            "encoding": "utf-8",
            "encoding_errors": "strict",
            "decode_responses": False,
        },
        connection_class=connection_class,
    )

    app.extensions["redis"] = redis_client
