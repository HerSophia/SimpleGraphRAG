import networkx as nx
from pyvis.network import Network
import random

class KnowledgeGraphVisualizer:
    def __init__(self, graph):
        """
        初始化知识图谱可视化器。
        :param graph: 知识图谱的内部结构（字典）。
        """
        self.graph = graph

    @staticmethod
    def random_color():
        """生成随机颜色的 HEX 代码."""
        return f"#{random.randint(0, 0xFFFFFF):06x}"

    def visualize(self, output_file="knowledge_graph.html"):
        """
        可视化知识图谱，并生成 HTML 文件。
        :param output_file: 输出的 HTML 文件名。
        """
        # 创建一个 NetworkX 图对象
        G = nx.DiGraph()

        # 遍历图谱，添加节点和边
        for entity, data in self.graph.items():
            # 随机颜色的节点
            node_color = self.random_color()
            G.add_node(entity, title=f"Type: {data['type']}", color=node_color, size=15)

            # 添加关系
            for relation_type, neighbors in data.get("relations", {}).items():
                for neighbor in neighbors:
                    target = neighbor["target"]
                    weight = neighbor["weight"]
                    edge_color = self.random_color()
                    G.add_edge(
                        entity,
                        target,
                        title=f"Relation: {relation_type}<br>Weight: {weight}",
                        label=relation_type,
                        weight=weight,
                        color=edge_color,
                        width=weight * 5  # 根据权重调整边的粗细
                    )

        # 使用 Pyvis 创建可交互的图
        net = Network(height="750px", width="100%", directed=True)
        net.from_nx(G)

        # 添加物理布局
        net.repulsion(
            node_distance=200,  # 节点之间的距离
            spring_length=200,  # 弹簧长度
            damping=0.9         # 阻尼系数
        )

        # 生成 HTML 文件
        net.show(output_file)
        print(f"知识图谱已保存为 HTML 文件：{output_file}")
