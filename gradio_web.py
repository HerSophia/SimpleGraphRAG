import asyncio
import re
import webbrowser
from cProfile import label

import gradio as gr
import json
from pathlib import Path
import httpx
import json
from collections import deque, OrderedDict, defaultdict
import heapq
import igraph as ig
import plotly.graph_objects as go
import random

from networkx.classes import DiGraph
from pyvis.network import Network

# 知识图谱文件路径
GRAPH_FILE = "knowledge_graph.json"
PATHS_FILE = "entity_paths.json"
HTML_FILE = "knowledge_graph.html"
API_KEY = "sk-0PXEjzN9ZX7lkRyQ39267329B2D54aB7Ae60B0D7647a5a14"



# 加载知识图谱
if Path(GRAPH_FILE).exists():
    with open(GRAPH_FILE, "r", encoding="utf-8") as f:
        knowledge_graph = json.load(f)
else:
    knowledge_graph = {}


# 异步提取实体和关系的函数
async def extract_entities_and_relations_with_weights(text, api_key):
    url = "https://www-my-office.rc6s3wcue.nyat.app:13332/v1/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    prompt = f"""
    从用户输入的文本（如果是JSON结构则按照结构中的数据来提取）中提取出所有的“实体”和它们之间的“关系”，并为每个关系分配一个重要性权重（范围为0到1）。使用以下纯文本格式返回结果，不要使用任何格式字符（例如粗体、斜体等），并且实体和关系描述中不要使用竖线 `|` 字符：

    实体：
    实体名称1|实体类型1
    实体名称2|实体类型2
    ...

    关系：
    实体名称1->实体名称2|关系类型|权重值
    实体名称3->实体名称4|关系类型|权重值
    ...

    文本：{text}
    """

    data = {
        "model": "gemini-1.5-pro",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 8192,
        "temperature": 0
    }

    timeout = httpx.Timeout(
        connect=10.0,
        read=60.0,
        write=10.0,
        pool=120.0
    )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            text_output = response.json()['choices'][0]['message']['content']

            print(f"LLM的原始输出:\n{text_output}")  # LLM的原始输出

            # 解析实体
            entities = []
            entity_match = re.findall(r"\s*\*?\*?实体：\s*\n(.*?)(?:\n\n|\Z)", text_output, re.DOTALL)
            if entity_match:  # 检查是否找到匹配项
                entity_lines = entity_match[0].strip().split('\n')
                for line in entity_lines:
                    name, type_ = line.split('|')
                    entities.append({"name": name.strip(), "type": type_.strip()})
            else:
                print("未找到实体信息")  # 输出调试信息

            # 解析关系
            relations = []
            relation_match = re.findall(r"\s*\*?\*?关系：\s*\n(.*?)(?:\n\n|\Z)", text_output, re.DOTALL)
            if relation_match:
                relation_lines = relation_match[0].strip().split('\n')
                for line in relation_lines:
                    parts = line.split('|', maxsplit=2)  # 最多分割成3部分，防止权重部分再次出现|
                    if len(parts) != 3:
                        print(f"关系行格式错误: {line}")
                        continue  # 跳过当前循环，处理下一行

                    source_target = parts[0]
                    relation = parts[1]
                    weight = parts[2]

                    source, target = source_target.split('->')
                    relations.append({
                        "source": source.strip(),
                        "target": target.strip(),
                        "relation": relation.strip(),
                        "weight": float(weight.strip())
                    })
            else:
                print("未找到关系信息")  # 输出调试信息
            return {"entities": entities, "relations": relations}

    except httpx.RequestError as e:
        print(f"HTTP Request Error: {e}")
        raise
    except httpx.HTTPStatusError as e:
        print(f"HTTP Status Error: {e.response.text}")
        raise
    except (ValueError, IndexError) as e: # 添加解析异常处理
        print(f"解析输出时发生错误: {e}, 输出文本为:\n{text_output}")
        raise # 重新抛出异常，方便上层处理
    except KeyError as e:
        print(f"KeyError: {e}, 原始响应为: {response.json()}")
        raise

def build_knowledge_graph(entities, relations):
    """
    构建知识图谱，将实体和关系组织为图的形式。
    :param entities: 包含实体信息的列表，每个实体包含 `name` 和 `type`。
    :param relations: 包含关系信息的列表，每个关系包含 `source`、`target`、`relation` 和 `weight`。
    :return: 构建的知识图谱数据结构。
    """
    knowledge_graph = {}

    # 初始化实体
    for entity in entities:
        name = entity["name"]
        etype = entity["type"]
        knowledge_graph[name] = {"type": etype, "relations": {}}

    # 添加关系
    for relation in relations:
        source = relation["source"]
        target = relation["target"]
        rel_type = relation["relation"]
        weight = relation["weight"]

        if source in knowledge_graph:
            if rel_type not in knowledge_graph[source]["relations"]:
                knowledge_graph[source]["relations"][rel_type] = []
            knowledge_graph[source]["relations"][rel_type].append({"target": target, "weight": weight})

    return knowledge_graph

def dfs_check_connection(graph, start, target, visited=None):
    """
    使用深度优先搜索检查两个实体之间是否存在路径。
    """
    if visited is None:
        visited = set()
    if start == target:
        return True
    visited.add(start)
    for relation, neighbors in graph.get(start, {}).get("relations", {}).items():
        for neighbor in neighbors:
            if neighbor["target"] not in visited:
                if dfs_check_connection(graph, neighbor["target"], target, visited):
                    return True
    return False

def bfs_shortest_path(graph, start, target):
    """
    使用广度优先搜索查找两个实体之间的最短路径。
    """
    queue = deque([(start, [start])])  # 队列存储当前节点及路径
    visited = set()  # 用于记录已访问的节点
    while queue:
        current, path = queue.popleft()
        if current == target:
            return path  # 找到目标节点，返回路径
        visited.add(current)
        for relation, neighbors in graph.get(current, {}).get("relations", {}).items():
            for neighbor in neighbors:
                if neighbor["target"] not in visited:
                    queue.append((neighbor["target"], path + [neighbor["target"]]))
    return None  # 无法找到路径时返回 None

def dijkstra(graph, start, target):
    """Dijkstra 算法，用于找到加权图中从起点到目标节点的最短路径。"""
    # 优先队列，存储元组 (累计权重, 当前节点, 当前路径)
    priority_queue = [(0, start, [start])]
    # 已访问的节点集合
    visited = set()

    while priority_queue:
        # 弹出优先队列中的权重最小的节点
        current_weight, current_node, path = heapq.heappop(priority_queue)

        # 如果当前节点已访问过，则跳过
        if current_node in visited:
            continue

        # 将当前节点标记为已访问
        visited.add(current_node)

        # 如果当前节点是目标节点，返回累计权重和路径
        if current_node == target:
            return current_weight, path

        # 遍历当前节点的所有邻居
        for relation, neighbors in graph.get(current_node, {}).get("relations", {}).items():
            for neighbor in neighbors:
                # 如果邻居节点未访问过
                if neighbor["target"] not in visited:
                    # 计算累积权重：将权重取反（1 - weight）以强调“重要性”
                    cumulative_weight = current_weight + (1 - neighbor["weight"])
                    # 将邻居节点加入优先队列
                    heapq.heappush(priority_queue,
                                   (cumulative_weight, neighbor["target"], path + [neighbor["target"]]))

    # 如果未找到路径，返回无穷大和空路径
    return float("inf"), []

def fix_illegal_json_content(content):
    """
    修复 JSON 字符串中包含的非法转义序列。
    :param content: 原始 JSON 字符串
    :return: 修复后的 JSON 字符串
    """
    # 替换非法的转义序列：\' 替换为 '
    content = re.sub(r"\\'", "'", content)

    # 替换非法的单个反斜杠，例如 \n 替换为 \\n，确保其合法性
    # 在合法反斜杠（如 \\）中间插入一个额外的反斜杠
    content = re.sub(r"(?<!\\)\\(?![\"\\/bfnrtu])", r"\\\\", content)

    return content

def format_response_data(response_data):
    # 格式化输出的函数，保持不变或根据需要修改
    output_string = "Entities:\n"
    for entity in response_data.get("entities", []):
        output_string += f"  - {entity['name']} ({entity['type']})\n"

    output_string += "\nRelations:\n"
    for relation in response_data.get("relations", []):
        output_string += f"  - {relation['source']} --{relation['relation']}--> {relation['target']} (Weight: {relation['weight']})\n"

    return output_string

def parse_llm_response(llm_response):
    try:
        # 现在的 llm_response 已经是 Python 字典，不需要再解析 JSON
        entities = llm_response.get("entities", [])
        relations = llm_response.get("relations", [])

        # 格式化输出
        output_data = format_response_data(llm_response) # 使用相同的格式化函数

        print(output_data)
        return entities, relations, output_data

    except (KeyError, IndexError, TypeError) as e: # 调整捕获的异常类型
        raise ValueError(f"Error parsing LLM response: {e}, 原始数据为: {llm_response}")

def save_graph_to_json(graph, filepath="knowledge_graph.json"):
    """
    将知识图谱保存为 JSON 文件，并还原为原始 JSON 结构。
    """
    entities = []
    relations = []

    for entity_name, data in graph.items():
        entities.append({"name": entity_name, "type": data["type"]})
        for relation, neighbors in data["relations"].items():
            for neighbor in neighbors:
                relations.append({
                    "source": entity_name,
                    "target": neighbor["target"],
                    "relation": relation,
                    "weight": neighbor["weight"]
                })

    json_data = {"entities": entities, "relations": relations}

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    return f"已保存知识图谱至{filepath}"

def save_paths_to_json(paths, filepath):
    """
    将计算出的路径数据保存为 JSON 文件。
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(paths, f, indent=2, ensure_ascii=False)

def load_json(filepath):
    """Load JSON data into an OrderedDict."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def parse_graph(data):
    """解析为合适的结构"""
    graph = defaultdict(lambda: {"relations": defaultdict(list)})
    for relation in data["relations"]:
        graph[relation["source"]]["relations"][relation["relation"]].append({
            "target": relation["target"],
            "weight": relation["weight"]
        })
    return graph

def compute_all_paths(graph):
    """计算图中所有实体对之间的最短路径。"""
    print(f"Graph type: {type(graph)}")
    print(f"Graph content: {graph}")

    all_paths = {}
    visited_pairs = set()  # 已访问过的实体对

    for start in graph.keys():
        print(f"Processing start entity: {start}")
        if start not in all_paths:
            all_paths[start] = {}

        for target in graph.keys():
            if start != target and (start, target) not in visited_pairs and (target, start) not in visited_pairs:
                print(f"Processing target entity: {target}")
                weight, path = dijkstra(graph, start, target)
                if path:  # 如果存在有效路径
                    all_paths[start][target] = {
                        "path": path,
                        "total_weight": weight
                    }
                    # 标记此实体对为已访问
                    visited_pairs.add((start, target))

    return all_paths

def find_relationships_between_entities(graph, paths_data, entity1, entity2):
    """
    查找两个实体之间的关系：
    - 直接连接：返回关系详情。
    - 间接连接：返回最短路径上的所有关系。
    - 无连接：返回无连接的提示信息。
    """
    print(graph)
    if entity1 not in graph or entity2 not in graph:
        return f"实体 '{entity1}' 或 '{entity2}' 不存在于图中。"

    # 检查直接连接
    for relation, neighbors in graph[entity1]["relations"].items():
        for neighbor in neighbors:
            if neighbor["target"] == entity2:
                return {
                    "type": "direct",
                    "relation": relation,
                    "details": neighbor
                }

    # 检查间接连接
    if entity1 in paths_data and entity2 in paths_data[entity1]:
        path = paths_data[entity1][entity2]["path"]
        connections = []
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            found_relation = False  # 标记是否找到关系
            if source in graph and "relations" in graph[source]: # 确保source存在于graph中且有relations
                for relation, neighbors in graph[source]["relations"].items():
                    for neighbor in neighbors:
                        if neighbor["target"] == target:
                            connections.append({
                                "source": source,
                                "target": target,
                                "relation": relation,
                                "details": neighbor
                            })
                            found_relation = True # 找到关系后标记为True并跳出内层循环
                            break
                    if found_relation:
                        break # 找到关系后跳出内层循环
            if not found_relation:  # 如果没有找到关系，则说明路径信息和图谱信息不一致
                print(f"警告：在 {source} 和 {target} 之间没有找到关系，路径信息可能不完整。")
        return {
            "type": "indirect",
            "path": path,
            "connections": connections
        }

    # 无连接
    return f"实体 '{entity1}' 和 '{entity2}' 之间没有连接。"

# user
def update_relationship(graph, entity1, entity2, new_relation):
    """
    更新两个实体之间的关系：
    - 如果直接关系存在，更新其关系。
    - 如果没有直接关系，则添加新关系。
    """
    if entity1 in graph:
        for relation, neighbors in graph[entity1]["relations"].items():
            for neighbor in neighbors:
                if neighbor["target"] == entity2:
                    neighbor.update(new_relation)
                    return f"更新了 '{entity1}' 和 '{entity2}' 之间的关系。"
        # 添加新的关系
        if "custom" not in graph[entity1]["relations"]:
            graph[entity1]["relations"]["custom"] = []
        graph[entity1]["relations"]["custom"].append({"target": entity2, **new_relation})
        return f"在 '{entity1}' 和 '{entity2}' 之间添加了新的关系。"
    return f"实体 '{entity1}' 不存在于图中。"

# llm
def add_or_update_entities_and_relations(graph, entities, relations):
    """在图中添加或更新实体和关系。"""
    for entity in entities:
        if entity["name"] not in graph:
            graph[entity["name"]] = {"type": entity["type"], "relations": {}}

    for relation in relations:
        source = relation["source"]
        target = relation["target"]
        rel_type = relation["relation"]
        weight = relation["weight"]

        if source in graph:
            if rel_type not in graph[source]["relations"]:
                graph[source]["relations"][rel_type] = []
            graph[source]["relations"][rel_type].append({"target": target, "weight": weight})

    return graph

def merge_graphs(graph1, graph2):
    """将两个图合并为一个图。"""
    for entity, data in graph2.items():
        if entity not in graph1:
            graph1[entity] = data
        else:
            # 合并关系
            for relation, neighbors in data["relations"].items():
                if relation not in graph1[entity]["relations"]:
                    graph1[entity]["relations"][relation] = neighbors
                else:
                    graph1[entity]["relations"][relation].extend(neighbors)
    return graph1

# 定义主要逻辑函数
async def process_text(input_text, api_key):
    llm_response = await extract_entities_and_relations_with_weights(input_text, api_key)
    entities, relations,output_data = parse_llm_response(llm_response)
    graph = build_knowledge_graph(entities, relations)
    return graph,output_data


def compute_the_paths_and_save(graph: dict, state):
    all_paths = compute_all_paths(graph)
    save_paths_to_json(all_paths, PATHS_FILE)
    return f"成功保存路径文件"


def find_relationship(graph, entity1, entity2):
    if not Path(PATHS_FILE).exists():
        return "请先计算路径后再查找关系。"

    with open(PATHS_FILE, "r", encoding="utf-8") as f:
        paths_data = json.load(f)
    #paths_data = json.loads(paths_data)
    print(paths_data)
    result = find_relationships_between_entities(graph, paths_data, entity1, entity2)
    return json.dumps(result, indent=2, ensure_ascii=False)

def merge_updated_graph(graph, new_graph):
    try:
        merged_graph = merge_graphs(graph, new_graph)
        save_graph_to_json(merged_graph, GRAPH_FILE)
        compute_the_paths_and_save(merged_graph)
        return merged_graph,f"合并成功"
    except Exception as e:
        return f"合并失败: {e}"

def generate_html(graph):
    visualize_igraph(graph,HTML_FILE)
    #visualize(graph, HTML_FILE)
    return f"HTML 文件生成于 {HTML_FILE}"

def random_color():
        """生成随机颜色的 HEX 代码."""
        return f"#{random.randint(0, 0xFFFFFF):06x}"

def visualize(graph,output_file="knowledge_graph.html"):
        """
        可视化知识图谱，并生成 HTML 文件。
        :param output_file: 输出的 HTML 文件名。
        """
        # 创建一个 NetworkX 图对象
        G = DiGraph()

        # 遍历图谱，添加节点和边
        for entity, data in graph.items():
            # 随机颜色的节点
            node_color = random_color()
            G.add_node(entity, title=f"Type: {data['type']}", color=node_color, size=15)

            # 添加关系
            for relation_type, neighbors in data.get("relations", {}).items():
                for neighbor in neighbors:
                    target = neighbor["target"]
                    weight = neighbor["weight"]
                    edge_color = random_color()
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

def visualize_igraph(graph, output_file="knowledge_graph.html"):
    g = ig.Graph(directed=True)

    vertices = list(graph.keys())
    g.add_vertices(vertices)

    edges = []
    edge_labels = []
    edge_weights = []
    edge_directions = []  # 存储边的方向：单向或双向

    # 构建边，并检测双向关系
    edge_set = set() # 用集合来存储已经存在的边，用于检测反向边
    for source_idx, (entity, data) in enumerate(graph.items()):
        for relation_type, neighbors in data.get("relations", {}).items():
            for neighbor in neighbors:
                try:
                    target_idx = vertices.index(neighbor["target"])
                    if graph[neighbor['target']]['relations'].get(relation_type) and 'relation' in \
                            graph[neighbor['target']]['relations'][relation_type][0]:
                        reverse_relation = graph[neighbor['target']]['relations'][relation_type][0]['relation']
                        edge_labels.append(f"{relation_type}/{reverse_relation}")
                        edge_directions.append("both")
                        edge_weights.append(neighbor["weight"])
                    else:
                        edges.append((source_idx, target_idx))
                        edge_labels.append(relation_type)
                        edge_weights.append(neighbor["weight"])
                        edge_directions.append("one")
                    edge = tuple(sorted((source_idx, target_idx)))
                    if edge not in edge_set:
                        edge_set.add(edge)

                except ValueError:
                    print(f"Target node '{neighbor['target']}' not found in graph.")

    g.add_edges(edges)

    visual_style = {}
    visual_style["vertex_size"] = 20
    visual_style["vertex_color"] = [random_color() for _ in range(g.vcount())]
    visual_style["edge_width"] = [w * 3 for w in edge_weights]
    visual_style["layout"] = g.layout_fruchterman_reingold()

    fig = go.Figure(data=[go.Scattergl(
        x=[visual_style["layout"][i][0] for i in range(g.vcount())],
        y=[visual_style["layout"][i][1] for i in range(g.vcount())],
        mode='markers+text',
        marker=dict(
            color=visual_style["vertex_color"],
            size=visual_style["vertex_size"]
        ),
        text=vertices,
        textposition="bottom center"
    )])

    for i, edge in enumerate(g.es):
        source, target = edge.tuple
        x0, y0 = visual_style["layout"][source]
        x1, y1 = visual_style["layout"][target]

        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2

        edge_color = random_color()  # 在循环内部生成颜色

        if edge_directions[i] == "both":
            fig.add_trace(go.Scattergl(x=[x0, x1], y=[y0, y1], mode='lines',
                                       line=dict(width=visual_style["edge_width"][i], color=edge_color),
                                       marker=dict(symbol="arrow-bar-up", size=10, color=edge_color)))
            fig.add_trace(go.Scattergl(x=[x1, x0], y=[y1, y0], mode='lines',
                                       line=dict(width=visual_style["edge_width"][i], color=edge_color),
                                       marker=dict(symbol="arrow-bar-down", size=10, color=edge_color)))
            my += 0.05
        else:
            fig.add_trace(go.Scattergl(x=[x0, x1], y=[y0, y1], mode='lines',
                                       line=dict(width=visual_style["edge_width"][i], color=edge_color),
                                       marker=dict(symbol="arrow", size=10, color=edge_color)))
            my += 0.02

        fig.add_trace(go.Scattergl(
            x=[mx], y=[my],
            mode='text',
            text=[edge_labels[i]],
            textposition='bottom center',
            textfont=dict(size=10),
            hoverinfo='none'
        ))

    fig.update_layout(
        title="Knowledge Graph",
        title_x=0.5,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        showlegend=False
    )
    fig.write_html(output_file)
    print(f"知识图谱已保存为 HTML 文件：{output_file}")

# 样式定义
css = """
    .green-button {
        background: linear-gradient(to right, #56ab2f, #a8e063);
        color: white;
        font-weight: bold;
    }
    .green-button:hover {
        background: linear-gradient(to right, #a8e063, #56ab2f);
    }
    """

# 界面定义
with gr.Blocks(css=css) as interface:
    with gr.Row():
        with gr.Column():
            input_textbox = gr.Textbox(label="输入文本", lines=30, placeholder="在此处输入您的文本...")
            result_textbox = gr.Textbox(label="回复/读取内容", lines=30, placeholder="大模型回复的内容...")
        with gr.Column():
            with gr.Row():
                send_button = gr.Button("发送文本", elem_classes=["green-button"])
            with gr.Row():
                save_graph_button = gr.Button("保存图谱文件", elem_classes=["green-button"])
                compute_paths_button = gr.Button("计算路径", elem_classes=["green-button"])

            with gr.Row():
                open_graph_button = gr.Button("打开 knowledge_graph.json", elem_classes=["green-button"])
                open_paths_button = gr.Button("打开 entity_paths.json", elem_classes=["green-button"])
            output_textbox2 = gr.Textbox(label="操作信息", visible=True)
            with gr.Row():
                read_graph_button = gr.Button("读取 knowledge_graph.json", elem_classes=["green-button"])
            with gr.Row():
                entity1_textbox = gr.Textbox(label="实体 1")
                entity2_textbox = gr.Textbox(label="实体 2")
            find_entities_relationship = gr.Button("查找两个实体之间的关系",elem_classes=["green-button"])
            relationship_result_textbox = gr.JSON(label="关系信息")
            with gr.Row():
                send_to_llm_button = gr.Button("发送编辑后的信息至大模型", elem_classes=["green-button"])
                merge_graph_button = gr.Button("合并新知识图谱", elem_classes=["green-button"])
            with gr.Row():
                generate_html_button = gr.Button("生成 HTML", elem_classes=["green-button"])
                open_html_button = gr.Button("打开 HTML", elem_classes=["green-button"])



    #output_textbox1 = gr.Textbox(visible=False)
    # 使用 State 来存储图数据
    output_graph_textbox1 = gr.State()
    output_graph_textbox2 = gr.State()
    output_json = gr.State()
    # 交互逻辑
    def processing_text(input_text):
        api_key = API_KEY
        data1,data2 = asyncio.run(process_text(input_text, api_key))
        print(data1)
        return data1,data2

    send_button.click(
        processing_text,
        inputs=input_textbox,
        outputs=[output_graph_textbox1, result_textbox]  # 用于显示生成的知识图谱 JSON
    )

    save_graph_button.click(
        save_graph_to_json,
        inputs=[output_graph_textbox1],
        outputs=output_textbox2
    )

    open_graph_button.click(
        lambda: Path(GRAPH_FILE).read_text(encoding="utf-8") if Path(GRAPH_FILE).exists() else "知识图谱文件不存在。",
        outputs=output_textbox2
    )

    open_paths_button.click(
        lambda: Path(PATHS_FILE).read_text(encoding="utf-8") if Path(PATHS_FILE).exists() else "路径文件不存在。",
        outputs=output_textbox2
    )

    compute_paths_button.click(
        compute_the_paths_and_save,
        inputs=[output_graph_textbox1],
        outputs=output_textbox2
    )

    def read_graph():
        with open(GRAPH_FILE, "r", encoding="utf-8") as f:
            read_graph = json.load(f)
        read_graph_str = json.dumps(read_graph,indent=4, ensure_ascii=False)
        #print(type(read_graph))
        entities, relations, output_data = parse_llm_response(read_graph)
        read_graph = build_knowledge_graph(entities, relations)
        #read_graph = Path(GRAPH_FILE).read_text(encoding="utf-8") if Path(GRAPH_FILE).exists() else "知识图谱文件不存在。",
        return read_graph_str,read_graph

    read_graph_button.click(
        fn=read_graph,
        outputs=[result_textbox,output_graph_textbox1]
    )

    find_entities_relationship.click(
        fn=find_relationship,
        inputs=[output_graph_textbox1, entity1_textbox, entity2_textbox],
        outputs=[output_json]
    )

    def str_to_json(data):
        return data

    output_json.change(
        fn=str_to_json,
        inputs=output_json,
        outputs=relationship_result_textbox
    )

    send_to_llm_button.click(
        processing_text,
        inputs=relationship_result_textbox,
        outputs=[output_graph_textbox2, result_textbox]
    )

    merge_graph_button.click(
        merge_updated_graph,
        inputs=[output_graph_textbox1,output_graph_textbox2],
        outputs=[output_graph_textbox1, result_textbox]
    )

    generate_html_button.click(
        generate_html,
        inputs=output_graph_textbox1,
        outputs=output_textbox2
    )

    def open_html():
        if Path(PATHS_FILE).exists():
            webbrowser.open(HTML_FILE)
            return f"已打开HTML文件"
        else:
            return f"HTML 文件不存在。"
    open_html_button.click(
        open_html,
        outputs=output_textbox2
    )

interface.launch()
