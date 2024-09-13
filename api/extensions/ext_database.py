from functools import wraps

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from databases import Database

POSTGRES_INDEXES_NAMING_CONVENTION = {
    "ix": "%(column_0_label)s_idx",
    "uq": "%(table_name)s_%(column_0_name)s_key",
    "ck": "%(table_name)s_%(constraint_name)s_check",
    "fk": "%(table_name)s_%(column_0_name)s_fkey",
    "pk": "%(table_name)s_pkey",
}

DATABASE_URL = "postgresql+asyncpg://postgres:difyai123456@localhost:5432/dify"
database = Database(DATABASE_URL)


metadata = MetaData(naming_convention=POSTGRES_INDEXES_NAMING_CONVENTION)
# Base = declarative_base(metadata=metadata)
#
db = SQLAlchemy(metadata=metadata)
def transactional(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        await database.connect()
        async with database.transaction():
            try:
                # Execute the function within the transaction
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                # If an exception occurs, transaction will automatically rollback
                print(f"Transaction failed and rolled back due to: {e}")
                raise e

    return wrapper


def init_app(app):
    db.init_app(app)
