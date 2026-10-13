"""The language model side: Jessica's agent and the helpers around it.

  agent.py       - the LangChain agent, its SQLite memory, and the reply stream
  interrupts.py  - keeps that memory honest when the user cuts a reply off
  llmtext.py     - read plain text out of LangChain message content
  llm_keys.py    - swap a new API key into already-built clients
"""
