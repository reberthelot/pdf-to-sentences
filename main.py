from __future__ import annotations

import uvicorn

from src.app.config import settings
from src.app.main import app

__all__ = ["app"]

if __name__ == "__main__":
    uvicorn.run("src.app.main:app", host=settings.HOST, port=settings.PORT, reload=True)

