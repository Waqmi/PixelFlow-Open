"""Stable application-facing API for the image engine.

The implementation remains in generate_main_images during the first migration
step. Keeping this small adapter gives the future Flutter client and the
current desktop UI the same entry point without changing image rules yet.
"""

from generate_main_images import RESOURCE_ROOT, SIZES, generate_images

__all__ = ["RESOURCE_ROOT", "SIZES", "generate_images"]
