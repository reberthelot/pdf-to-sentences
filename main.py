from backend import app as backend_app
from fastapi.responses import RedirectResponse
from frontend import app as ui_app

# Point root application to backend API
app = backend_app

# Mount UI sub-application onto /ui route
app.mount("/ui", ui_app)


@app.get("/", include_in_schema=False)
def root_redirect() -> RedirectResponse:
    """Redirect root browser requests to the UI application."""
    return RedirectResponse(url="/ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

