import random

from fedot.core.visualisation.pipeline_specific_visuals import PipelineVisualizer
from golem.core.adapter import DirectAdapter
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.visualisation.opt_viz import PlotTypesEnum
from golem.visualisation.opt_viz_extra import visualise_pareto
from matplotlib import pyplot as plt

from nas.graph.base_graph import NasGraph
from nas.graph.node.nas_graph_node import NasNode
from nas.graph.node.nas_node_params import NasNodeFactory
from nas.repository.layer_types_enum import LayersPoolEnum
from nas.utils.utils import project_root

# path = project_root() / "_results/debug/master_2/2024-08-09_23-30-56/history.json"
# path = r"D:\dev\aim\nas_kan_results\_results\smaller-kans-mnist\history.json"
path = r"C:\dev\aim\nas_kan_results\_results\cifar-thorough-cnn-cifar10\history.json"
# path = r"C:\dev\aim\nas_kan_results\_results\torch-kan-dense-kan-cifar10\history.json"

history = OptHistory.load(path)

print(history)

# Analyze if there is a graph in the history that has arity 2 at least for one node:
for generation in history.individuals:
    for ind in generation:
        graph = ind.graph
        for node in graph.nodes:
            if len(node.nodes_from) >= 2:
                print(f"Arity 2 node found in graph: {graph}, {node.uid}")
                break

# Check if some of the nodes are kan_linear:
from visualize_comparisons import get_individual_dump

linear_count = 0

for ind in get_individual_dump(history):
    graph = ind.graph
    for node in graph.nodes:
        if node.content["name"] == "kan_linear":
            print(f"KANLinear node found in graph: {graph}, {node.uid}")
            linear_count += 1
            break

        if node.content["name"] == "linear":
            print(f"Linear node found in graph: {graph}, {node.uid}")
            linear_count += 1
            break

print(f"Percentage of graphs containing linear layers: {linear_count / len(get_individual_dump(history))}")

# Plot hist of total parameter count or flops for all individuals in the history:
values = []
for generation in history.individuals:
    for ind in generation:
        values.append(ind.fitness.values[1])
        # values.append(DirectAdapter(base_graph_class=NasGraph, base_node_class=NasNode).restore(
        #     ind.graph).get_singleton_node_by_name('flatten').content["total_model_flops"])

from seaborn import histplot, distplot, displot

histplot(values, log_scale=True, bins=5)
plt.show()
# displot(values, log_scale=True, bins=5)
# plt.show()

# history.show(PlotTypesEnum.operations_animated_bar)
# history.show(PlotTypesEnum.operations_kde)
# history.show(PlotTypesEnum.fitness_line_interactive)

visualise_pareto(
    history.final_choices.data, show=True, objectives_names=("LogLoss", "Parameters")
)

print(history.get_leaderboard())


def print_model_layers(graph):
    node = graph.root_node

    layers = []
    while True:
        # print(node.content)
        layers.append(node.content)
        assert len(node.nodes_from) <= 1
        if not node.nodes_from:
            break
        node = node.nodes_from[0]

    return layers


def generate_model_code(layer_specs, num_classes):
    """
    Given a list of layer dictionaries (from a printed model‐graph)
    and a number of classification classes, this function returns a string
    of complete PyTorch code defining an nn.Module with:
      - A features block (nn.Sequential) that stacks layers, including conv/kan layers,
        batch normalization (if "momentum" and "epsilon" are provided),
        activations, and pooling.
      - A final classifier (Linear) whose input dimension is determined by
        passing an example input through the features.
      - Utilities to compute and print the final flattened features,
        parameter counts, and the spatial geometry (channels, height, width)
        of the last convolutional layer.
      
    The function assumes that layer_specs is given in output-to-input order,
    so we reverse it to obtain input->output order for the forward pass.
    
    For "conv2d" layers, a standard nn.Conv2d is generated.
    For "kan_conv2d" layers, a special KANConv2DLayer (imported from your_kan_module)
    is used.
    
    If a layer’s 'params' contain keys for "momentum" and "epsilon",
    a BatchNorm layer is added immediately following the convolution.
    """
    # Reverse the layer list to have input->output order
    layers_in_order = list(reversed(layer_specs))
    layer_lines = []  # will build the nn.Sequential definition
    final_flatten_dim = None  # to capture the flattened dimension from a flatten layer

    # A simple activation mapping – extend as needed.
    activation_map = {
        'relu': "nn.ReLU()",
        'sigmoid': "nn.Sigmoid()",
        'tanh': "nn.Tanh()",
        'elu': "nn.ELU()",
        'softsign': "nn.Softsign()",
        'softplus': "nn.Softplus()",
    }

    for layer in layers_in_order:
        lname = layer.get("name")

        if lname in ["conv2d", "kan_conv2d"]:
            # Common parameters
            in_channels = layer["dims"]["input"]["channels"]
            out_channels = layer["params"]["out_shape"]
            kernel_size = layer["params"]["kernel_size"]
            stride = layer["params"].get("stride", 1)
            padding = layer["params"].get("padding", "same")
            # Use repr so that lists are rendered correctly (e.g., [2, 2])
            padding_str = repr(padding)
            kernel_size_str = repr(kernel_size)

            if lname == "conv2d":
                conv_line = (f"nn.Conv2d({in_channels}, {out_channels}, "
                             f"kernel_size={kernel_size_str}, stride={stride}, padding={padding_str})")
            else:
                # For KAN, additional parameters (grid_size, spline_order) must be provided.
                grid_size = layer["params"]["grid_size"]
                spline_order = layer["params"]["spline_order"]
                conv_line = (f"KANConv2DLayer(input_dim={in_channels}, output_dim={out_channels}, "
                             f"kernel_size={kernel_size_str}, stride={stride}, padding={padding_str}, "
                             f"groups=1, dilation=1, grid_size={grid_size}, spline_order={spline_order})")
            layer_lines.append(conv_line)

            # If momentum and epsilon are provided, add a BN layer.
            if "momentum" in layer["params"] and "epsilon" in layer["params"]:
                momentum = layer["params"]["momentum"]
                epsilon = layer["params"]["epsilon"]
                bn_line = f"nn.BatchNorm2d({out_channels}, momentum={momentum}, eps={epsilon})"
                layer_lines.append(bn_line)

            # If activation is provided, add here.
            act = layer["params"].get("activation")
            if act and act.lower() in activation_map:
                layer_lines.append(activation_map[act.lower()])

            # If pooling is specified, add pooling layer.
            if "pooling_kernel_size" in layer["params"]:
                pool_kernel = layer["params"]["pooling_kernel_size"]
                pool_mode = layer["params"].get("pooling_mode", "max").lower()
                if pool_mode == "max":
                    layer_lines.append(f"nn.MaxPool2d(kernel_size={pool_kernel})")
                elif pool_mode == "average":
                    layer_lines.append(f"nn.AvgPool2d(kernel_size={pool_kernel})")

        elif lname == "flatten":
            # Record flatten dimension if available
            final_flatten_dim = layer["dims"].get("output_dim", layer["dims"].get("flattened"))
            layer_lines.append("nn.Flatten()")

        else:
            # Add other custom layer types if needed
            pass

    # Fallback: if no flatten layer is provided, try to compute the flattened dimension
    if final_flatten_dim is None:
        last = layers_in_order[-1]
        if "dims" in last and "output" in last["dims"]:
            out_dims = last["dims"]["output"]
            final_flatten_dim = out_dims["channels"] * (out_dims["side_size"] ** 2)
        else:
            final_flatten_dim = "unknown_dim"

    # Build the classifier layer definition.
    classifier_line = f"nn.Linear(flattened_features, num_classes)"

    layer_text = ",\n".join(f"            {line}" for line in layer_lines)
    final_code = f"""import torch
import torch.nn as nn

# If using KAN layers, adjust the import to match your package structure
from your_kan_module import KANConv2DLayer

class GeneratedModel(nn.Module):
    def __init__(self, num_classes, input_channels=3, input_geometry=(32, 32)):
        super(GeneratedModel, self).__init__()
        self.features = nn.Sequential(
{layer_text}
        )

        # Create a dummy input to compute output dimensions
        example_input = torch.randn(1, input_channels, *input_geometry)
        
        # Determine convolutional output geometry before flattening if possible
        modules = list(self.features.children())
        if modules and isinstance(modules[-1], nn.Flatten):
            conv_features = nn.Sequential(*modules[:-1])
        else:
            conv_features = self.features
        conv_output = conv_features(example_input)
        
        # Flattened features size for the classifier:
        flattened_features = self.features(example_input).shape[1]
        self.classifier = {classifier_line}
        
        # Utility to count parameters
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        conv_params = count_parameters(self.features)
        classifier_params = count_parameters(self.classifier)
        total_params = count_parameters(self)
        
        print(f"Using flattened features: {{flattened_features}}; conv parameters: {{conv_params}}; classifier parameters: {{classifier_params}}; total: {{total_params}}")
        
        # Print last convolutional layer geometry if output is 4D
        if conv_output.dim() == 4:
            # conv_output shape: [batch, channels, height, width]
            channels, height, width = conv_output.shape[1], conv_output.shape[2], conv_output.shape[3]
            print(f"Last conv layer output size: channels={{channels}}, height={{height}}, width={{width}}")
        else:
            print(f"Unexpected conv output shape: {{conv_output.shape}}")
        
    def forward(self, x):
        if train is not None:
            if train:
                self.train()
            else:
                self.eval()
        x = self.features(x)
        x = self.classifier(x)
        return x"""

    return final_code

def example_generate():
    cnn_layers = [
        {'name': 'flatten',
         'params': {},
         'parameter_count': 0,
         'dims': {'dim_kind': 'flatten', 'input': {'channels': 256, 'side_size': 4},
                  'flattened': 4096, 'output_dim': 4096},
         'total_model_parameter_count': 3136000},
        {'name': 'conv2d',
         'params': {'out_shape': 256, 'kernel_size': 5, 'activation': 'sigmoid', 'stride': 1,
                    'padding': 1, 'momentum': 0.99, 'epsilon': 0.001,
                    'pooling_kernel_size': 2, 'pooling_mode': 'max',
                    'output_node_grid_size': 10, 'output_node_spline_order': 3},
         'parameter_count': 819968,
         'dims': {'dim_kind': '2d',
                  'input': {'channels': 128, 'side_size': 8},
                  'weighted_layer_output_shape': {'channels': 256, 'side_size': 8},
                  'output': {'channels': 256, 'side_size': 4}}},
        # ... more layers ...
    ]

    model_code = generate_model_code(cnn_layers, num_classes=10)
    print(model_code)


def visualize_model_structures():
    for i, chosen in enumerate(history.final_choices):
        graph = chosen.graph
        # Add info to nodes:
        for node in graph.nodes:
            print(node.content)
            if node.name == "flatten":
                continue
            params = node.content["params"]
            if "conv" in node.name:
                ks = params["kernel_size"]
                filters = params["out_shape"]
                new_name = f"{node.name}[{ks}×{ks}, {filters}]"
            else:
                dims = params["out_shape"]
                new_name = f"{node.name}[{dims}]"
            node.content["name"] = new_name
        # plt.figure(figsize=(30, 30))
        PipelineVisualizer(chosen.graph).visualise(node_size_scale=0.7, font_size_scale=0.5, save_path=f"../../tmp/pipeline-{i}.png", dpi=1000)
        print(generate_model_code(print_model_layers(chosen.graph), num_classes=10))
        print("========================")


def check_skip_connections():
    print("Checking skip connections...")
    for ind in get_individual_dump(history):
        graph = ind.graph
        for node in graph.nodes:
            if len(node.nodes_from) == 2:
                print(f"Skip connection found in graph: {graph}, {node.uid}")

if __name__ == '__main__':
    visualize_model_structures()
    # check_skip_connections()
