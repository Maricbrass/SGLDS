"""Database configuration and connection."""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://sglds_user:sglds_password@localhost:5432/sglds_db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,  # Avoid connection pool issues in async context
    echo=os.getenv("DEBUG", "False").lower() == "true"
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base for declarative models
Base = declarative_base()

def get_db():
    """Dependency for FastAPI to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
