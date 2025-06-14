from flask import Flask, redirect, url_for
import os
from .graph_builder import build_pipeline_graph

app = Flask(__name__, static_folder='static')

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
REF_PLAN_DIR = os.path.join(BASE_DIR, 'refactor_plan')
OUTPUT_HTML = os.path.join(app.static_folder, 'pipeline.html')


def generate_graph():
    build_pipeline_graph(REF_PLAN_DIR, OUTPUT_HTML)


@app.route('/')
def index():
    if not os.path.exists(OUTPUT_HTML):
        generate_graph()
    with open(OUTPUT_HTML, 'r', encoding='utf-8') as f:
        return f.read()


@app.route('/refresh')
def refresh():
    generate_graph()
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
