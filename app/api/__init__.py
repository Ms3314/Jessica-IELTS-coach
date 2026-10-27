"""The web server's surface: what the browser can call.

  ids.py     - validation patterns for ids that arrive from the browser
  routes.py  - REST routes (settings, sessions, profile, exams)
  voice.py   - the /ws WebSocket that carries a live call

server.py at the project root builds the FastAPI app from these.
"""



# usually we make these init.pys so that we can import as modules from different places 