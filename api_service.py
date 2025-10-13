import pymysql
import requests
import decimal
from elasticsearch import Elasticsearch
import json

# Query API
class QueryService:
    def __init__(self):
        self.host = "127.0.0.1"
        self.username = "root"
        self.password = "<your_password>"
        self.database = "chaiys"

    def query(self, sql):
        global conn
        try:
            # 连接MySQL获取表结构
            conn = pymysql.connect(
                host=self.host,
                port=3306,
                user=self.username,
                password=self.password,
                database=self.database
            )
            cursor = conn.cursor()
            cursor.execute(sql)

            # 返回字段列表
            return cursor.fetchall()
        finally:
            conn.close()

    def query_with_column(self, sql):
        global conn
        try:
            # 连接MySQL获取表结构
            conn = pymysql.connect(
                host=self.host,
                user=self.username,
                password=self.password,
                database=self.database
            )
            cursor = conn.cursor()
            cursor.execute(sql)

            # 获取字段名称
            columns = [col[0] for col in cursor.description]
            # 获取数据
            data = cursor.fetchall()

            # 将数据转换为字典列表格式
            # 将数据转换为字典列表格式
            result = []
            for row in data:
                # 处理每个字段的值
                processed_row = []
                for value in row:
                    if isinstance(value, decimal.Decimal):
                        # 转换为 float 或 str
                        processed_row.append(float(value))  # 或 str(value)
                    else:
                        processed_row.append(value)
                result.append(dict(zip(columns, processed_row)))

            return result
        finally:
            conn.close()

# Semantics API
class SemanticService:
    def __init__(self):
        self.es_client = Elasticsearch(
            hosts=["http://127.0.0.1:9200"],
            basic_auth=("elastic", "<your_password>"),
            verify_certs=False
        )
        # 向量化模型
        self.ollama_host = "http://localhost:11434/api/embeddings"
        self.metadata_index = "metadata_index"
        self.modal_name = "llama2"
        self.mapping = {
            "mappings": {
                "properties": {
                    "table_info": {
                        "type": "text",
                        "analyzer": "ik_max_word",  # 中文分词
                        "search_analyzer": "ik_smart"
                    },
                    "nomic_embedding": {
                        "type": "dense_vector",
                        "dims": 4096,
                        "index": True,
                        "similarity": "cosine",
                    }
                }
            }
        }

    def get_embedding(self, text):
        """使用 Ollama 生成文本嵌入向量"""
        try:
            print(f"调用大模型llama2向量化：{text}")
            response = requests.post(self.ollama_host, json={"model": self.modal_name, "prompt": text})
            response.raise_for_status()
            embedding = response.json()["embedding"]
            return embedding
        except Exception as e:
            raise RuntimeError(f"嵌入生成失败: {str(e)}")

    def delete_index(self):
        """安全删除索引"""
        try:
            if self.es_client.indices.exists(index=self.metadata_index):
                self.es_client.indices.delete(index=self.metadata_index)
                print(f"索引 {self.metadata_index} 删除成功")
                return True
            return None
        except Exception as e:
            print(f"删除索引失败: {type(e).__name__}: {str(e)}")
            return False

    def create_index(self):
        self.delete_index()
        """创建支持nomic向量的索引"""
        self.es_client.indices.create(index=self.metadata_index, body=self.mapping)
        print(f"索引 {self.metadata_index} 创建成功")

    def vectorize_and_index(self, prompt, content):
        """生成文本嵌入向量并插入索引"""
        doc = {
            "table_info": content,
            "nomic_embedding": self.get_embedding(prompt)
        }
        self.es_client.index(index=self.metadata_index, document=doc)
        self.es_client.indices.refresh(index=self.metadata_index)
        print(f"表信息 {prompt}:{content} 向量化成功")

    def semantic_search(self, user_query, k):
        """执行语义相似度搜索"""
        query_embedding = self.get_embedding(user_query)

        knn_query = {
            "knn": {
                "field": "nomic_embedding",
                "query_vector": query_embedding,
                "k": k,
                "num_candidates": 100  # 这就是ef_search参数
            },
            "_source": ["table_info"]  # 返回原始问题
        }

        response = self.es_client.search(index=self.metadata_index, body=knn_query)
        table_info = [
            {
                "score": hit["_score"],
                "table_info": hit["_source"]["table_info"],
                "id": hit["_id"]
            }
            for hit in response["hits"]["hits"]
        ]
        print(f"自然语言语义检索字段成功，匹配到的元数据信息：{table_info}")
        return table_info

    def keyword_search(self, user_query: str, k) -> list:
        """基于分词的关键词匹配搜索"""
        search_query = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "table_info": {
                                    "query": user_query,
                                    "analyzer": "ik_smart",  # 使用IK中文分词器
                                    "boost": 1.0
                                }
                            }
                        },
                        {
                            "match_phrase": {
                                "table_info": {
                                    "query": user_query,
                                    "slop": 2,  # 允许短语间隔
                                    "boost": 0.5
                                }
                            }
                        }
                    ]
                }
            },
            "size": k,
            "_source": ["table_info"],
            "highlight": {
                "fields": {
                    "table_info": {}  # 返回高亮片段
                }
            }
        }

        response = self.es_client.search(index=self.metadata_index, body=search_query)

        table_info = [
            {
                "score": hit["_score"],
                "table_info": hit["_source"]["table_info"],
                "id": hit["_id"],
                "highlight": hit.get("highlight", {}).get("table_info", [])
            }
            for hit in response["hits"]["hits"]
        ]
        print(f"自然语言分词搜索字段成功，匹配到的元数据信息：{table_info}")
        return table_info

    def hybrid_search(self, user_query: str, k, alpha: float = 0.7) -> list:
        """混合搜索（语义+关键词）"""
        # 语义搜索
        semantic_results = self.semantic_search(user_query, k * 2)
        semantic_map = {hit["id"]: hit for hit in semantic_results}

        # 关键词搜索
        keyword_results = self.keyword_search(user_query, k * 2)
        keyword_map = {hit["id"]: hit for hit in keyword_results}

        # 合并结果
        all_ids = set(semantic_map.keys()) | set(keyword_map.keys())
        combined = []

        for doc_id in all_ids:
            semantic_score = semantic_map.get(doc_id, {}).get("score", 0)
            keyword_score = keyword_map.get(doc_id, {}).get("score", 0)

            combined.append({
                "id": doc_id,
                "table_info": semantic_map.get(doc_id, keyword_map.get(doc_id))["table_info"],
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "combined_score": alpha * semantic_score + (1 - alpha) * keyword_score,
                "highlight": keyword_map.get(doc_id, {}).get("highlight", [])
            })

        # 按综合分数排序
        combined.sort(key=lambda x: x["combined_score"], reverse=True)

        table_info = combined[:k]
        print(f"自然语言混合检索字段成功，匹配到的元数据信息：{table_info}")
        return table_info

# Analysis API
class AnalysisService:
    def __init__(self):
        self.ollama_host = "http://localhost:11434/api/chat"

    def analysis(self, prompt, model="deepseek-r1:32b"):
        # 发送POST请求
        str = ""
        # 请求数据
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": True
        }
        with requests.post(self.ollama_host, json=data, stream=True) as response:
            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        # 解析JSON数据
                        chunk = json.loads(decoded_line)
                        str += chunk['message']['content']
                        # 打印消息内容
                        print(chunk['message']['content'], end='', flush=True)
                    except json.JSONDecodeError:
                        print(f"无法解析JSON: {decoded_line}")
        return str