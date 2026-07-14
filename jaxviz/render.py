"""Rendering layer: serialize the graph data into the bundled HTML template and
display or export the interactive graph.
"""
import json
import uuid
from pathlib import Path
from string import Template
from importlib import resources

from IPython.display import display, HTML

from .enums import ExportFormat


def generate_html_file_action(html_str, unique_id, export_path=None):
    if export_path is not None:
        base_path = Path(export_path).expanduser()
        if base_path.suffix:
            output_file = base_path
        else:
            output_file = base_path / f'jaxviz_graph_{unique_id}.html'
    else:
        output_file = Path.cwd() / f'jaxviz_graph_{unique_id}.html'

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html_str, encoding='utf-8')
    resolved_output = output_file.resolve()
    display(HTML(f"""
        <div id="jaxviz-container-{unique_id}" style="font-family: Arial, sans-serif; margin: 12px 0;">
            <div style="font-size: 14px; color: #333;">
                <b>Saved as <code>{resolved_output}</code></b>
            </div>
        </div>
    """))
    return resolved_output


def plot_graph(adj_list, module_info, func_info, node_to_module_path,
               parent_module_to_nodes, parent_module_to_depth,
               graph_node_name_to_without_suffix, graph_node_display_names,
               node_to_attr_name, ancestor_map, collapse_modules_after_depth,
               height, width, export_format, show_module_attr_names,
               repeat_containers, show_modular_view=False, export_path=None):
    unique_id = str(uuid.uuid4())
    template_str = resources.read_text('jaxviz.templates', 'graph.html')
    d3_source = resources.read_text('jaxviz.assets', 'd3.min.js')
    viz_source = resources.read_text('jaxviz.assets', 'viz-standalone.js')
    jsoneditor_css = resources.read_text('jaxviz.assets', 'jsoneditor-10.2.0.min.css')
    jsoneditor_source = resources.read_text('jaxviz.assets', 'jsoneditor-10.2.0.min.js')

    template = Template(template_str)

    output = template.safe_substitute({
        'adj_list_json': json.dumps(adj_list),
        'module_info_json': json.dumps(module_info),
        'func_info_json': json.dumps(func_info),
        'parent_module_to_nodes_json': json.dumps(parent_module_to_nodes),
        'parent_module_to_depth_json': json.dumps(parent_module_to_depth),
        'graph_node_name_to_without_suffix': json.dumps(graph_node_name_to_without_suffix),
        'graph_node_display_names': json.dumps(graph_node_display_names),
        'node_to_attr_name': json.dumps(node_to_attr_name),
        'ancestor_map': json.dumps(ancestor_map),
        'repeat_containers': json.dumps(list(repeat_containers)),
        'unique_id': unique_id,
        'd3_source': d3_source,
        'viz_source': viz_source,
        'jsoneditor_css': jsoneditor_css,
        'jsoneditor_source': jsoneditor_source,
        'collapse_modules_after_depth': collapse_modules_after_depth,
        'node_to_module_path': node_to_module_path,
        'show_module_attr_names': 'true' if show_module_attr_names else 'false',
        'height': f'{height}px' if (export_format not in (ExportFormat.PNG, ExportFormat.SVG)) else '0px',
        'width': f'{width}px' if width is not None else '100%',
        'generate_image': 'true' if export_format is ExportFormat.PNG else 'false',
        'generate_svg': 'true' if export_format is ExportFormat.SVG else 'false',
        'show_modular_view': 'true' if show_modular_view else 'false',
    })

    if export_format == ExportFormat.HTML:
        return generate_html_file_action(output, unique_id, export_path=export_path)
    else:
        display(HTML(output))
        return None


def validate_export_format(export_format):
    if export_format is None:
        return None
    export_format = export_format.lower()
    valid_values = [e.value for e in ExportFormat]
    if export_format not in valid_values:
        raise ValueError(
            f"Invalid export format: {export_format}. Must be one of {valid_values}."
        )
    return ExportFormat(export_format)
