"""Console entry point for the cropper API."""
import uvicorn

from .config import Settings


def main() -> None:
    # Built here rather than using the module singleton so the process reads the
    # environment it was actually started with.
    config = Settings()
    uvicorn.run(
        "v_cropper.service.app:app",
        host=config.bind_host,
        port=config.bind_port,
        log_level=config.log_level,
    )


if __name__ == "__main__":
    main()
