"""Dependency injection container — wires all services together."""

from src.common.di.container import Container, create_container

__all__ = [
    "Container",
    "create_container",
]
