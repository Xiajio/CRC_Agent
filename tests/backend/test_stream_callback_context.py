from contextvars import Context

from src.nodes.node_utils import clear_stream_callback, set_stream_callback


def test_clear_stream_callback_ignores_token_from_closed_context():
    token = Context().run(set_stream_callback, lambda _payload: None)

    clear_stream_callback(token)
