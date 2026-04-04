# AGENTS.md

## AI Behavior and Rules

1. AI can automatically fix bugs when root cause is clear.
2. AI must **NOT delete existing files**.
3. AI should suggest improvements for readability, structure, and performance.
4. AI must ask before major structural changes that could alter business logic.
5. Do not modify original files under `/data`; create derived files elsewhere when needed.
6. Keep modules separated: training, inference, and web app should stay decoupled.
7. Follow PEP8 style and add comments for important logic.
