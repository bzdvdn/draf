# Teff CLI demo: run, trace, and graph the hello_workflow example.
Output docs/assets/hello-workflow-cli.gif
Set Width 1000
Set Height 520
Set FontSize 16
Set Padding 24
Set PlaybackSpeed 1.4

Type "source .venv/bin/activate"
Sleep 500ms
Enter
Sleep 1.5s

Type "teff run --file examples/hello_workflow/workflow.yaml"
Sleep 500ms
Enter
Sleep 3s

Type "teff run --file examples/hello_workflow/workflow.yaml --trace"
Sleep 500ms
Enter
Sleep 6s

Type "teff graph examples/hello_workflow/workflow.yaml"
Sleep 500ms
Enter
Sleep 6s