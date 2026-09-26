"""Render compatibility entrypoint.

Render currently uses: python bot.py
The actual HTTP/webhook server lives in render_server.py.
"""

from render_server import main


if __name__ == "__main__":
    main()
