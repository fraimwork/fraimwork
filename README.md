# Fraimwork

A tool that enables intelligent interaction with framework-based repositories.

When working with LLMs like GitHub Copilot, ChatGPT, and Google Gemini, a concept that comes up frequently is caching larger context windows that are repeatedly used for generation. This is especially relevant in the event of a debug loop, where an agentic system is zeroing in on a particular buggy snippet of code. Dependencies (and maybe even dependents) do not need to be repeatedly computed by the LLM and can have their contexts cached. The question is: what system design would ensure that such caching remains efficient and effective? When can we afford to evict from cache? Can such a system scale effectively, and if so, what system design techniques would be needed to ensure this?
