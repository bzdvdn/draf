# Teff LLM demo: a streaming ReAct agent, then a YAML workflow with --trace.
Output docs/assets/react-agent-cli.gif
Set Width 1000
Set Height 520
Set FontSize 16
Set Padding 24
Set PlaybackSpeed 1.2

Type "source .venv/bin/activate"
Sleep 500ms
Enter
Sleep 1.5s

Type "python examples/react_agent/main.py"
Sleep 500ms
Enter
Sleep 9s

Type "teff run --file examples/branching/workflow.yaml --trace"
Sleep 500ms
Enter
Sleep 8s