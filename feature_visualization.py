import html
import ipywidgets as widgets
from IPython.display import display

def render_activation_html(feature_id, activating_examples):
    css = """
    <style>
        .example-container {
            font-family: sans-serif;
            margin-bottom: 20px;
            border: 1px solid #444;
            background: #1e1e1e;
            color: #ddd;
            border-radius: 6px;
            overflow: hidden;
        }
        .header {
            background: #333;
            padding: 8px 12px;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .score-pill {
            padding: 4px 10px;
            border-radius: 12px;
            font-family: monospace;
            font-size: 0.9em;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }
        .content {
            padding: 15px;
            overflow-wrap: break-word;
            white-space: pre-wrap;
            background: #1a1a1a;
        }
        .txt {
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
            line-height: 1.5;
        }
    </style>
    """

    html_content = [css]
    html_content.append(f"<h3>Feature {feature_id} - Top Activating Examples</h3>")

    sorted_examples = sorted(activating_examples, key=lambda x: x["activation"], reverse=True)

    max_activation = max([ex["activation"] for ex in sorted_examples], default=1.0)
    if max_activation == 0:
        max_activation = 1.0

    for i, example in enumerate(sorted_examples):
        activation = example["activation"]
        intensity = activation / max_activation
        background = f"rgba(0, 150, 255, {max(0.3, intensity)})"

        html_content.append(f"""
        <div class="example-container">
            <div class="header">
                <span>Rank #{i + 1}</span>
                <span class="score-pill" style="background: {background}; color: white;">
                    Activation: {activation:.4f}
                </span>
            </div>
            <div class="content">
                <div class="txt">{html.escape(example['text'])}</div>
            </div>
        </div>
        """)

    return "".join(html_content)


def visualize_features(loader, feature_id_list, k=10):
    current_index = [0]

    feature_label = widgets.Label()
    html_output = widgets.HTML()

    def show():
        feature_id = feature_id_list[current_index[0]]
        feature_label.value = (
            f"  Feature: {feature_id} ({current_index[0] + 1}/{len(feature_id_list)})  "
        )

        if not loader.has_sufficient_activating_examples(feature_id, k):
            html_output.value = (
                f"<pre>Skipping feature {feature_id}: "
                f"insufficient data (needs at least {k} examples).</pre>"
            )
            return

        activating_examples = loader.get_activating_examples_split(feature_id, k, 0)
        html_output.value = render_activation_html(feature_id, activating_examples)

    button_next = widgets.Button(description="Next", icon="arrow-right")

    def on_next(_):
        current_index[0] = (current_index[0] + 1) % len(feature_id_list)
        show()

    button_next.on_click(on_next)

    button_previous = widgets.Button(description="Previous", icon="arrow-left")

    def on_previous(_):
        current_index[0] = (current_index[0] - 1) % len(feature_id_list)
        show()

    button_previous.on_click(on_previous)

    controls = widgets.HBox(
        [button_previous, feature_label, button_next],
        layout=widgets.Layout(align_items="center", margin="0 0 20px 0")
    )

    display(widgets.VBox([controls, html_output]))
    show()