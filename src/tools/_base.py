import logging

logger = logging.getLogger(__name__)

_registry = []


def register_tool(cls):
    _registry.append(cls)
    return cls


def get_registered_tools():
    return list(_registry)


class BaseTool:
    def __init__(self, handler):
        self.handler = handler

    @property
    def audio(self):
        return self.handler.audio

    @property
    def osc(self):
        return self.handler.osc

    @property
    def tracker(self):
        return self.handler.tracker

    @property
    def wanderer(self):
        return self.handler.wanderer

    @property
    def personality(self):
        return self.handler.personality

    @property
    def config(self):
        return self.handler.config

    @property
    def session(self):
        return self.handler.session

    @property
    def live_session(self):
        return self.handler.live_session

    def declarations(self, config=None):
        return []

    async def handle(self, name, args):
        return None
