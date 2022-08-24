from ._tf_graph_builders.graph_builders import TFGraphBuilderFactory
from ._tf_graph_builders_1x.graph_builders import TFGraphBuilderFactory as TFGraphBuilderFactory1x

tf_graph = TFGraphBuilderFactory()
tf_graph_1x = TFGraphBuilderFactory1x()
