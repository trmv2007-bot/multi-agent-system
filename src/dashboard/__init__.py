"""Live web dashboard: a chat interface over the agent team.

Nothing is imported here on purpose. ``python -m src.dashboard.server`` imports
this package first, and an eager import of the server module from inside it is
what produces runpy's "found in sys.modules" warning at every start. Import from
``src.dashboard.server`` directly instead.
"""

__all__: list[str] = []
