# Pipeline Viewer

This folder contains a simple Flask web application that visualizes the
function call pipeline of the modules inside `refactor_plan`.

Run the app with:

```bash
python -m pipeline_viewer.app
```

Open `http://localhost:5000` in your browser to see the graph.
Click the **Refresh** link to update the pipeline after modifying code
under `refactor_plan`.
