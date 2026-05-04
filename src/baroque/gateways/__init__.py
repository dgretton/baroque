"""Inference gateway implementations."""

from collections.abc import Mapping

from baroque.core.interfaces import InferenceGateway
from baroque.core.models import ProviderRequest, ProviderResponse
from baroque.gateways.anthropic_native import AnthropicGateway
from baroque.gateways.openai_compatible import OpenAICompatibleGateway


class MultiplexingGateway:
    """Route a `ProviderRequest` to the gateway registered for its provider."""

    def __init__(self, gateways: Mapping[str, InferenceGateway]) -> None:
        if not gateways:
            raise ValueError("MultiplexingGateway requires at least one gateway")
        self._gateways: dict[str, InferenceGateway] = dict(gateways)

    async def send(self, request: ProviderRequest) -> ProviderResponse:
        gateway = self._gateways.get(request.provider)
        if gateway is None:
            registered = ", ".join(sorted(self._gateways)) or "(none)"
            raise ValueError(
                f"no gateway registered for provider: {request.provider} "
                f"(registered: {registered})"
            )
        return await gateway.send(request)


__all__ = [
    "AnthropicGateway",
    "MultiplexingGateway",
    "OpenAICompatibleGateway",
]
