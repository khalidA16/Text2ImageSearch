
from typing import Any


class Embedder():
    def __init__(self) -> None:
        pass
        
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

class ImageEmbedder(Embedder):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)

class TextEmbedder(Embedder):
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
    