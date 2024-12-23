import httpx
import json
from collections import deque, OrderedDict, defaultdict
import heapq



# OpenAI API Request Function
def extract_entities_and_relations_with_weights(text, api_key):
    url = "https://mysite:13332/v1"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Prompt to extract entities, relationships, and their weights
    prompt = f"""
    从以下文本（如果是JSON结构则按照结构中的数据来提取）中提取出所有的“实体”和它们之间的“关系”，并为每个关系分配一个重要性权重（范围为0到1），并以 JSON 格式返回：
    输出格式：
    {
        "entities": [
            {"name": "实体名称", "type": "实体类型"}
        ],
        "relations": [
            {"source": "实体名称1", "target": "实体名称2", "relation": "关系类型", "weight": 权重值}
        ]
    }

    文本：{text}
    """

    data = {
        "model": "gemini-pro-1.5",
        "prompt": prompt,
        "max_tokens": 8192,
        "temperature": 0
    }

    with httpx.Client() as client:
        response = client.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

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

def parse_llm_response(llm_response):
    try:
        response_data = json.loads(llm_response["choices"][0]["text"])
        entities = response_data.get("entities", [])
        relations = response_data.get("relations", [])
        return entities, relations
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        raise ValueError(f"Error parsing LLM response: {e}")

def save_graph_to_json(graph, filepath):
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
    all_paths = {}
    visited_pairs = set()  # 已访问过的实体对

    for start in graph.keys():
        if start not in all_paths:
            all_paths[start] = {}

        for target in graph.keys():
            if start != target and (start, target) not in visited_pairs and (target, start) not in visited_pairs:
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
            for relation, neighbors in graph[source]["relations"].items():
                for neighbor in neighbors:
                    if neighbor["target"] == target:
                        connections.append({
                            "source": source,
                            "target": target,
                            "relation": relation,
                            "details": neighbor
                        })
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


# Main execution
if __name__ == "__main__":
    text_input = "亚历山大是马其顿王国的国王，他曾征服了埃及，并建立了亚历山大城。"
    api_key = "sk-0PXEjzN9ZX7lkRyQ39267329B2D54aB7Ae60B0D7647a5a14"

    try:
        # Extract entities, relations, and weights
        result = extract_entities_and_relations_with_weights(text_input, api_key)
        print("Response from OpenAI:")
        print(json.dumps(result, indent=2))

        # Parse entities and relations from response
        llm_response = extract_entities_and_relations_with_weights(text_input, api_key)
        entities, relations = parse_llm_response(llm_response)

        # Build knowledge graph
        knowledge_graph = build_knowledge_graph(entities, relations)
        print("\nGenerated Knowledge Graph:")
        print(json.dumps(knowledge_graph, indent=2, ensure_ascii=False))

        # Save knowledge graph to file
        save_graph_to_json(knowledge_graph, "knowledge_graph.json")

        # Load knowledge graph and paths from JSON files
        knowledge_graph = load_json("knowledge_graph.json")
        paths_data = load_json("entity_paths.json")

        knowledge_graph = parse_graph(knowledge_graph)


        # Compute and save all paths between entities
        paths = compute_all_paths(knowledge_graph)
        print("\nComputed Paths:")
        print(json.dumps(paths, indent=2, ensure_ascii=False))
        save_paths_to_json(paths, "entity_paths.json")



        # Find relationships between entities
        entity1 = "亚历山大"
        entity2 = "埃及"
        result = find_relationships_between_entities(knowledge_graph, paths_data, entity1, entity2)

        print("\nRelationship Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Example of updating a relationship by user
        updated_relation = {"relation": "new_relation", "weight": 0.8}
        update_message = update_relationship(knowledge_graph, entity1, entity2, updated_relation)
        print(update_message)

        # Example of updating a graph by using data from other graph

        # Save the updated graph
        save_graph_to_json(knowledge_graph,"knowledge_graph.json")

    except Exception as e:
        print(f"Error: {e}")
