import json
from collections import deque


class KnowledgeGraphManager:
    import httpx
    import json
    from collections import deque, defaultdict
    import heapq

    def __init__(self, api_key="sk-0PXEjzN9ZX7lkRyQ39267329B2D54aB7Ae60B0D7647a5a14", api_url="https://www-my-office.rc6s3wcue.nyat.app:13332/v1"):
        self.api_key = api_key
        self.api_url = api_url


    def extract_entities_and_relations_with_weights(self, text):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        prompt = f"""
        从以下文本（如果是JSON结构则按照结构中的数据来提取）中提取出所有的“实体”和它们之间的“关系”，
        并为每个关系分配一个重要性权重（范围为0到1），并以 JSON 格式返回：
        输出格式：
        {{
            "entities": [
                {{"name": "实体名称", "type": "实体类型"}}
            ],
            "relations": [
                {{"source": "实体名称1", "target": "实体名称2", "relation": "关系类型", "weight": 权重值}}
            ]
        }}

        文本：{text}
        """
        data = {
            "model": "gemini-pro-1.5",
            "prompt": prompt,
            "max_tokens": 8192,
            "temperature": 0
        }

        try:
            with self.httpx.Client() as client:
                response = client.post(self.api_url, headers=headers, json=data)
                response.raise_for_status()
                return response.json()
        except self.httpx.RequestError as e:
            raise ValueError(f"HTTP Request failed: {e}")

    @staticmethod
    def parse_llm_response(llm_response):
        try:
            response_data = json.loads(llm_response["choices"][0]["text"])
            entities = response_data.get("entities", [])
            relations = response_data.get("relations", [])
            return entities, relations
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            raise ValueError(f"Error parsing LLM response: {e}")


    @staticmethod
    def build_knowledge_graph(entities, relations):
        knowledge_graph = {}
        for entity in entities:
            name = entity["name"]
            etype = entity["type"]
            knowledge_graph[name] = {"type": etype, "relations": {}}

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

    @staticmethod
    def save_graph_to_json(graph, filepath):
        entities = [{"name": name, "type": data["type"]} for name, data in graph.items()]
        relations = [
            {"source": entity_name, "target": neighbor["target"], "relation": rel, "weight": neighbor["weight"]}
            for entity_name, data in graph.items()
            for rel, neighbors in data["relations"].items()
            for neighbor in neighbors
        ]
        json_data = {"entities": entities, "relations": relations}

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def save_paths_to_json(paths, filepath):
        """
        将计算出的路径数据保存为 JSON 文件。
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(paths, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_json(filepath):
        """Load JSON data into an OrderedDict."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def load_graph_from_json(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = self.json.load(f)
        graph = self.defaultdict(lambda: {"relations": self.defaultdict(list)})
        for relation in data["relations"]:
            graph[relation["source"]]["relations"][relation["relation"]].append({
                "target": relation["target"],
                "weight": relation["weight"]
            })
        return graph


    def dijkstra(self, graph, start, target):
        priority_queue = [(0, start, [start])]
        visited = set()

        while priority_queue:
            current_weight, current_node, path = self.heapq.heappop(priority_queue)
            if current_node in visited:
                continue
            visited.add(current_node)
            if current_node == target:
                return current_weight, path

            for relation, neighbors in graph.get(current_node, {}).get("relations", {}).items():
                for neighbor in neighbors:
                    if neighbor["target"] not in visited:
                        cumulative_weight = current_weight + (1 - neighbor["weight"])
                        self.heapq.heappush(priority_queue,
                                            (cumulative_weight, neighbor["target"], path + [neighbor["target"]]))
        return float("inf"), []

    def compute_all_paths(self,graph: dict) -> dict:
        all_paths = {}
        visited_pairs = set()
        for start in graph.keys():
            all_paths[start] = {}
            for target in graph.keys():
                if start != target and (start, target) not in visited_pairs:
                    weight, path = self.dijkstra(graph, start, target)
                    if path:
                        all_paths[start][target] = {"path": path, "total_weight": weight}
                        visited_pairs.add((start, target))
        return all_paths

    def dfs_check_connection(self, graph, start, target, visited=None):
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
                    if self.dfs_check_connection(graph, neighbor["target"], target, visited):
                        return True
        return False

    def bfs_shortest_path(self, graph, start, target):
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



    @staticmethod
    def find_relationships_between_entities(graph, paths_data, entity1, entity2):
        if entity1 not in graph or entity2 not in graph:
            return f"实体 '{entity1}' 或 '{entity2}' 不存在于图中。"

        for relation, neighbors in graph[entity1]["relations"].items():
            for neighbor in neighbors:
                if neighbor["target"] == entity2:
                    return {"type": "direct", "relation": relation, "details": neighbor}

        if entity1 in paths_data and entity2 in paths_data[entity1]:
            path = paths_data[entity1][entity2]["path"]
            connections = [
                {"source": path[i], "target": path[i + 1], "relation": rel, "details": neighbor}
                for i in range(len(path) - 1)
                for rel, neighbors in graph[path[i]]["relations"].items()
                for neighbor in neighbors if neighbor["target"] == path[i + 1]
            ]
            return {"type": "indirect", "path": path, "connections": connections}

        return f"实体 '{entity1}' 和 '{entity2}' 之间没有连接。"

    # user
    @staticmethod
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
    def add_or_update_entities_and_relations(self, graph, entities, relations):
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

    @staticmethod
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

    def update_entity_type(self, graph, entity_name, new_type):
        """
        更新图中某个实体的类型。

        :param graph: 知识图谱
        :param entity_name: 实体名称
        :param new_type: 新的实体类型
        """
        if entity_name in graph:
            graph[entity_name]["type"] = new_type
            return f"实体 '{entity_name}' 的类型已更新为 '{new_type}'。"
        else:
            return f"实体 '{entity_name}' 不存在于图中。"


    def add_or_update_relation(self, graph, source, target, relation, weight):
        """
        添加或更新图中两个实体之间的关系。

        :param graph: 知识图谱
        :param source: 源实体名称
        :param target: 目标实体名称
        :param relation: 关系类型
        :param weight: 关系权重（0-1）
        """
        if source not in graph:
            graph[source] = {"type": "未知类型", "relations": {}}
        if relation not in graph[source]["relations"]:
            graph[source]["relations"][relation] = []
        for neighbor in graph[source]["relations"][relation]:
            if neighbor["target"] == target:
                neighbor["weight"] = weight  # 更新权重
                return f"已更新 '{source}' 和 '{target}' 之间关系 '{relation}' 的权重为 {weight}。"
        # 如果没有现有关系，添加新关系
        graph[source]["relations"][relation].append({"target": target, "weight": weight})
        return f"已添加 '{source}' 和 '{target}' 之间的新关系 '{relation}'，权重为 {weight}。"


    def delete_entity_or_relation(self, graph, entity_name=None, source=None, target=None, relation=None):
        """
        删除实体或实体之间的关系。

        :param graph: 知识图谱
        :param entity_name: 要删除的实体名称（如果指定则删除整个实体及其关系）
        :param source: 源实体名称（如果指定则删除关系）
        :param target: 目标实体名称（如果指定则删除关系）
        :param relation: 关系类型（如果指定则删除关系）
        """
        if entity_name:
            if entity_name in graph:
                del graph[entity_name]  # 删除实体及其关系
                # 删除其他实体中与该实体相关的所有关系
                for entity, data in graph.items():
                    for rel_type in list(data["relations"].keys()):
                        data["relations"][rel_type] = [
                            rel for rel in data["relations"][rel_type] if rel["target"] != entity_name
                        ]
                        if not data["relations"][rel_type]:  # 删除空的关系类型
                            del data["relations"][rel_type]
                return f"实体 '{entity_name}' 及其相关关系已删除。"
            else:
                return f"实体 '{entity_name}' 不存在于图中。"

        if source and target and relation:
            if source in graph and relation in graph[source]["relations"]:
                original_length = len(graph[source]["relations"][relation])
                graph[source]["relations"][relation] = [
                    rel for rel in graph[source]["relations"][relation] if rel["target"] != target
                ]
                if not graph[source]["relations"][relation]:  # 如果关系类型为空，则删除该类型
                    del graph[source]["relations"][relation]
                if len(graph[source]["relations"].keys()) == 0:
                    del graph[source]["relations"]
                if len(graph[source]["relations"].get(relation, [])) < original_length:
                    return f"已删除 '{source}' 和 '{target}' 之间的关系 '{relation}'。"
        return f"指定的关系或实体不存在。"

'''
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
'''