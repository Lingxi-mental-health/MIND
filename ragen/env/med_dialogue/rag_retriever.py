"""
RAG 检索模块 - 基于 BGE 嵌入向量进行诊疗指南/文献检索

流程：
1. 加载预计算的 BGE 嵌入向量
2. 对生成的 query 进行编码
3. 通过向量相似度检索最相关的证据文本
"""

import json
import numpy as np
import torch
from typing import List, Tuple, Optional
import os

# 路径根目录：默认相对本仓库，可用环境变量覆盖，避免把内部基础设施路径写进代码
DATA_ROOT = os.environ.get("MIND_DATA_ROOT", "data")



def _load_json_sidecar(path: str, what: str, default=None):
    """安全加载 JSON 侧车文件，取代 pickle.load。

    历史索引以 .pkl 存放，反序列化等同执行任意代码；此处只接受 .json，
    遇到遗留 .pkl 则拒绝并给出转换指引（见 scripts/convert_legacy_rag_index.py）。
    ``default`` 为 None 时文件缺失视为致命错误，否则返回 ``default``。
    """
    json_path = path[:-4] + ".json" if path.endswith(".pkl") else path
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    legacy_path = json_path[:-5] + ".pkl"
    if os.path.exists(legacy_path):
        raise RuntimeError(
            f"{what}: found a legacy pickle index at {legacy_path}. Unpickling would "
            f"deserialise arbitrary Python objects, so it is refused. Convert it with:\n"
            f"  python scripts/convert_legacy_rag_index.py --path {legacy_path} "
            f"--i-trust-this-file"
        )

    if default is None:
        raise FileNotFoundError(f"{what}: neither {json_path} nor a converted index exists")
    print(f"[RAGRetriever] {what} not found at {json_path}, using empty default")
    return default


class RAGRetriever:
    """
    基于 BGE 嵌入向量的 RAG 检索器
    
    用于将问诊过程中生成的 query 检索语句与预计算的证据库进行匹配，
    返回最相关的诊疗指南/文献证据。
    """
    
    _instance = None  # 单例模式，避免重复加载
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        embeddings_path: str = os.path.join(DATA_ROOT, "embeddings/retrieval_query_bge_large_zh_v1.5.npy"),
        texts_path: str = os.path.join(DATA_ROOT, "embeddings/retrieval_query_texts.pkl"),
        reasoning_support_path: str = os.path.join(DATA_ROOT, "embeddings/retrieval_query_reasoning_support.pkl"),
        reasoning_scores_path: str = os.path.join(DATA_ROOT, "embeddings/retrieval_query_reasoning_scores.pkl"),
        index_mapping_path: str = os.path.join(DATA_ROOT, "embeddings/retrieval_query_index_mapping.pkl"),
        bge_model_path: str = os.environ.get("MIND_BGE_MODEL_PATH", "BAAI/bge-large-zh-v1.5"),
        device: str = None,
        top_k: int = 3
    ):
        """
        初始化 RAG 检索器
        
        Args:
            embeddings_path: 预计算的嵌入向量路径
            texts_path: 原始文本路径（状态语句）
            reasoning_support_path: 推理支持文本路径
            reasoning_scores_path: 推理质量分数路径（1-5分）
            index_mapping_path: 索引映射路径
            bge_model_path: BGE 模型本地路径
            device: 计算设备
            top_k: 默认返回的 top-k 结果数
        """
        if self._initialized:
            return
        
        self.embeddings_path = embeddings_path
        self.texts_path = texts_path
        self.reasoning_support_path = reasoning_support_path
        self.reasoning_scores_path = reasoning_scores_path
        self.index_mapping_path = index_mapping_path
        self.bge_model_path = bge_model_path
        self.top_k = top_k
        
        # 确定设备
        # 【优化】默认使用 CPU 进行 BGE 编码，避免和 vLLM/Actor 争 GPU 显存
        # CPU 编码单条 query 约 10-30ms，可接受；且不会造成 GPU 抖动
        if device is None:
            self.device = "cpu"  # 强制 CPU，避免 GPU 争用
        else:
            self.device = device
        
        # 延迟加载
        self.embeddings = None
        self.texts = None
        self.reasoning_supports = None  # 推理支持文本
        self.reasoning_scores = None  # 推理质量分数（1-5分）
        self.index_mapping = None
        self.sentence_model = None  # 使用 sentence-transformers
        
        self._initialized = True
        print(f"[RAGRetriever] Initialized (lazy loading enabled)")
    
    def _load_embeddings(self):
        """加载预计算的嵌入向量"""
        if self.embeddings is not None:
            return
        
        print(f"[RAGRetriever] Loading embeddings from {self.embeddings_path}")
        # allow_pickle=False：.npy 只应承载数值数组，拒绝内嵌 Python 对象
        self.embeddings = np.load(self.embeddings_path, allow_pickle=False)
        print(f"[RAGRetriever] Embeddings loaded: shape={self.embeddings.shape}")
        
        # 归一化嵌入向量（用于余弦相似度计算）
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-8)
    
    def _load_texts(self):
        """加载原始文本（状态语句）"""
        if self.texts is not None:
            return
        
        print(f"[RAGRetriever] Loading texts from {self.texts_path}")
        self.texts = _load_json_sidecar(self.texts_path, "texts")
        print(f"[RAGRetriever] Texts loaded: count={len(self.texts)}")
    
    def _load_reasoning_supports(self):
        """加载推理支持文本"""
        if self.reasoning_supports is not None:
            return
        
        print(f"[RAGRetriever] Loading reasoning supports from {self.reasoning_support_path}")
        self.reasoning_supports = _load_json_sidecar(
            self.reasoning_support_path, "reasoning supports", default=[]
        )
        print(f"[RAGRetriever] Reasoning supports loaded: count={len(self.reasoning_supports)}")
    
    def _load_reasoning_scores(self):
        """加载推理质量分数（1-5分）"""
        if self.reasoning_scores is not None:
            return
        
        print(f"[RAGRetriever] Loading reasoning scores from {self.reasoning_scores_path}")
        self.reasoning_scores = _load_json_sidecar(
            self.reasoning_scores_path, "reasoning scores", default=[]
        )
        print(f"[RAGRetriever] Reasoning scores loaded: count={len(self.reasoning_scores)}")
    
    def _load_index_mapping(self):
        """加载索引映射"""
        if self.index_mapping is not None:
            return
        
        print(f"[RAGRetriever] Loading index mapping from {self.index_mapping_path}")
        self.index_mapping = _load_json_sidecar(self.index_mapping_path, "index mapping")
        print(f"[RAGRetriever] Index mapping loaded: count={len(self.index_mapping)}")
    
    def _load_sentence_model(self):
        """加载 sentence-transformers 模型用于编码 query"""
        if self.sentence_model is not None:
            return
        
        print(f"[RAGRetriever] Loading sentence-transformers model from: {self.bge_model_path}")
        try:
            from sentence_transformers import SentenceTransformer
            
            # 使用本地路径加载模型
            self.sentence_model = SentenceTransformer(
                self.bge_model_path,
                device=self.device,
                local_files_only=True
            )
            print(f"[RAGRetriever] Sentence model loaded on {self.device}")
        except Exception as e:
            print(f"[RAGRetriever] Failed to load sentence model: {e}")
            raise
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        使用 sentence-transformers 对 query 进行编码
        
        Args:
            query: 检索 query 文本
            
        Returns:
            编码后的向量 (1, dim)
        """
        self._load_sentence_model()
        
        # 编码并归一化
        embedding = self.sentence_model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embedding.reshape(1, -1)
    
    def encode_queries_batch(self, queries: List[str], batch_size: int = 32) -> np.ndarray:
        """
        批量编码多个 query
        
        Args:
            queries: query 列表
            batch_size: 批处理大小
            
        Returns:
            编码后的向量 (n, dim)
        """
        self._load_sentence_model()
        
        # 使用 sentence-transformers 批量编码
        embeddings = self.sentence_model.encode(
            queries,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False
        )
        
        return embeddings
    
    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, float, int, str, int]]:
        """
        检索与 query 最相关的证据文本
        
        Args:
            query: 检索 query 文本
            top_k: 返回的 top-k 结果数
            
        Returns:
            List of (text, similarity_score, original_index, reasoning_support, quality_score)
            - quality_score: 推理质量分数 (1-5)，0 表示未知
        """
        if top_k is None:
            top_k = self.top_k
        
        # 确保所有数据已加载
        self._load_embeddings()
        self._load_texts()
        self._load_reasoning_supports()
        self._load_reasoning_scores()
        self._load_index_mapping()
        
        # 编码 query
        query_embedding = self.encode_query(query)
        
        # 计算余弦相似度
        similarities = np.dot(self.embeddings, query_embedding.T).squeeze()
        
        # 【优化】使用 argpartition 替代 argsort，复杂度从 O(N log N) 降到 O(N)
        # argpartition 返回的是无序的 top-k 索引，需要再对这 k 个元素排序
        if top_k < len(similarities):
            # 找出相似度最大的 top_k 个索引（无序）
            top_k_unsorted = np.argpartition(similarities, -top_k)[-top_k:]
            # 对这 k 个索引按相似度排序（只排 k 个，很快）
            top_indices = top_k_unsorted[np.argsort(similarities[top_k_unsorted])[::-1]]
        else:
            # 如果 top_k >= N，直接全排序
            top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            text = self.texts[idx]
            score = float(similarities[idx])
            original_idx = self.index_mapping[idx] if self.index_mapping else idx
            # 获取推理支持文本
            reasoning_support = ""
            if self.reasoning_supports and idx < len(self.reasoning_supports):
                reasoning_support = self.reasoning_supports[idx] or ""
            # 获取质量分数
            quality_score = 0
            if self.reasoning_scores and idx < len(self.reasoning_scores):
                quality_score = self.reasoning_scores[idx] or 0
            results.append((text, score, original_idx, reasoning_support, quality_score))
        
        return results
    
    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = None
    ) -> List[List[Tuple[str, float, int, str, int]]]:
        """
        批量检索多个 query
        
        Args:
            queries: query 列表
            top_k: 每个 query 返回的 top-k 结果数
            
        Returns:
            每个 query 的检索结果列表，每条结果为 (text, score, original_idx, reasoning_support, quality_score)
            - quality_score: 推理质量分数 (1-5)，0 表示未知
        """
        if top_k is None:
            top_k = self.top_k
        
        if not queries:
            return []
        
        # 确保所有数据已加载
        self._load_embeddings()
        self._load_texts()
        self._load_reasoning_supports()
        self._load_reasoning_scores()
        self._load_index_mapping()
        
        # 批量编码
        query_embeddings = self.encode_queries_batch(queries)
        
        # 计算相似度矩阵
        similarities = np.dot(query_embeddings, self.embeddings.T)
        
        # 获取每个 query 的 top-k 结果
        all_results = []
        n_embeddings = similarities.shape[1]
        for i in range(len(queries)):
            # 【优化】使用 argpartition 替代 argsort
            if top_k < n_embeddings:
                top_k_unsorted = np.argpartition(similarities[i], -top_k)[-top_k:]
                top_indices = top_k_unsorted[np.argsort(similarities[i, top_k_unsorted])[::-1]]
            else:
                top_indices = np.argsort(similarities[i])[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                text = self.texts[idx]
                score = float(similarities[i, idx])
                original_idx = self.index_mapping[idx] if self.index_mapping else idx
                # 获取推理支持文本
                reasoning_support = ""
                if self.reasoning_supports and idx < len(self.reasoning_supports):
                    reasoning_support = self.reasoning_supports[idx] or ""
                # 获取质量分数
                quality_score = 0
                if self.reasoning_scores and idx < len(self.reasoning_scores):
                    quality_score = self.reasoning_scores[idx] or 0
                results.append((text, score, original_idx, reasoning_support, quality_score))
            
            all_results.append(results)
        
        return all_results
    
    def format_evidence(self, results: List, max_length: int = 500) -> str:
        """
        将检索结果格式化为证据文本
        
        Args:
            results: 检索结果列表，支持 (text, score, idx) 或 (text, score, idx, reasoning_support)
            max_length: 每条证据的最大长度
            
        Returns:
            格式化的证据文本
        """
        if not results:
            return "（未检索到相关证据）"
        
        evidence_parts = []
        for i, item in enumerate(results, 1):
            # 兼容 3 元组或 4 元组
            text = item[0]
            score = float(item[1])
            # 截断过长的文本
            if len(text) > max_length:
                text = text[:max_length] + "..."
            evidence_parts.append(f"[证据{i}] (相似度: {score:.3f})\n{text}")
        
        return "\n\n".join(evidence_parts)
    
    def retrieve_top4_for_categories(self, query: str, max_length: int = 300) -> str:
        """
        检索 top-4 结果，分别标记为 4 个类别的参考证据。
        
        由于目前没有按类别分组的证据库，这里使用 top-4 相似度最高的结果，
        并将其标记为 Depression/Anxiety/Mix/Others 对应的参考（便于模型理解）。
        
        Args:
            query: 检索 query
            max_length: 每条证据的最大长度
            
        Returns:
            格式化的 4 条证据文本
        """
        results = self.retrieve(query, top_k=4)
        
        if not results:
            return "（未检索到相关证据）"
        
        category_labels = ["Depression（抑郁）", "Anxiety（焦虑）", "Mix（混合）", "Others（其他）"]
        evidence_parts = []
        
        for i, item in enumerate(results):
            if i >= 4:
                break
            # 兼容 3 元组或 4 元组
            text = item[0]
            score = float(item[1])
            category = category_labels[i] if i < len(category_labels) else f"类别{i+1}"
            # 截断过长的文本
            if len(text) > max_length:
                text = text[:max_length] + "..."
            evidence_parts.append(f"【{category}】(相似度: {score:.3f})\n{text}")
        
        return "\n\n".join(evidence_parts)
    
    def retrieve_by_category(self, query: str, top_k: int = 10) -> dict:
        """
        按类别检索证据（每个类别返回最相关的 1 条）。
        
        Args:
            query: 检索 query
            top_k: 初始检索的 top-k 数量
            
        Returns:
            dict: {category_name: {"text": str, "score": float}, ...}
        """
        category_evidence, _ = self.retrieve_by_category_with_scores(query, top_k)
        return category_evidence
    
    def retrieve_by_category_with_scores(
        self, query: str, top_k: int = 10
    ) -> Tuple[dict, List[float]]:
        """
        按类别检索证据，同时返回相似度分数、推理支持和质量分数。
        
        由于目前没有按类别分组的证据库，这里使用 top-4 相似度最高的结果，
        并将其分配给 Depression/Anxiety/Mix/Others 四个类别。
        
        Args:
            query: 检索 query
            top_k: 初始检索的 top-k 数量
            
        Returns:
            Tuple[dict, List[float]]:
                - dict: {category_name: {"text": str, "score": float, "reasoning_support": str, "quality_score": int}, ...}
                - List[float]: 每个类别的相似度分数列表
        """
        results = self.retrieve(query, top_k=top_k)
        
        category_names = ["Depression", "Anxiety", "Mix", "Others"]
        category_evidence = {}
        similarity_scores = []
        
        # 使用 top-4 结果分配给 4 个类别
        for i, category in enumerate(category_names):
            if i < len(results):
                text, score, _, reasoning_support, quality_score = results[i]
                category_evidence[category] = {
                    "text": text, 
                    "score": score,
                    "reasoning_support": reasoning_support,
                    "quality_score": quality_score
                }
                similarity_scores.append(score)
            else:
                category_evidence[category] = None
        
        return category_evidence, similarity_scores
    
    def format_evidence_by_category(self, category_evidence: dict, max_length: int = 300) -> str:
        """
        将按类别检索的证据格式化为文本（旧版，按类别标签展示）。
        
        Args:
            category_evidence: {category_name: {"text": str, "score": float}, ...}
            max_length: 每条证据的最大长度
            
        Returns:
            格式化的证据文本
        """
        if not category_evidence:
            return "（未检索到相关证据）"
        
        category_labels = {
            "Depression": "Depression（抑郁）",
            "Anxiety": "Anxiety（焦虑）",
            "Mix": "Mix（混合）",
            "Others": "Others（其他）"
        }
        
        evidence_parts = []
        for category, info in category_evidence.items():
            if info is None:
                continue
            
            text = info.get("text", "")
            score = info.get("score", 0.0)
            label = category_labels.get(category, category)
            
            # 截断过长的文本
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            evidence_parts.append(f"【{label}】(相似度: {score:.3f})\n{text}")
        
        if not evidence_parts:
            return "（未检索到相关证据）"
        
        return "\n\n".join(evidence_parts)
    
    def format_evidence_for_inquiry(self, category_evidence: dict, max_state_length: int = 350, max_support_length: int = 400) -> str:
        """
        将检索到的证据格式化为【状态语句】+【推理支持】格式，供问诊参考。
        
        新格式不再按类别标签（Depression/Anxiety 等）展示，而是：
        - 【状态语句】：相似案例的状态描述（retrieval_query）
        - 【推理支持】：该案例对应的推理分析（reasoning_support）
        
        Args:
            category_evidence: {category_name: {"text": str, "score": float, "reasoning_support": str}, ...}
            max_state_length: 每条状态语句的最大长度
            max_support_length: 每条推理支持的最大长度
            
        Returns:
            格式化的证据文本
        """
        if not category_evidence:
            return "（未检索到相关证据）"
        
        evidence_parts = []
        evidence_idx = 1
        has_reasoning_support_count = 0  # 统计有推理支持的条目数
        
        for category, info in category_evidence.items():
            if info is None:
                continue
            
            text = info.get("text", "")
            score = info.get("score", 0.0)
            reasoning_support = info.get("reasoning_support", "")
            quality_score = info.get("quality_score", 0)  # 质量分数 (1-5)
            
            # 统计有推理支持的条目
            if reasoning_support and len(reasoning_support.strip()) > 10:
                has_reasoning_support_count += 1
            
            # 截断过长的推理支持
            if reasoning_support and len(reasoning_support) > max_support_length:
                reasoning_support = reasoning_support[:max_support_length] + "..."
            
            # 构建证据条目 - 只保留推理支持，不展示状态语句（状态语句仅用于检索匹配）
            if reasoning_support and reasoning_support.strip():
                # 有推理支持文本，展示推理支持
                # 格式：【参考案例 N】(相似度: X.XX, 质量: Y/5)
                evidence_parts.append(
                    f"【参考案例 {evidence_idx}】(相似度: {score:.2f}, 质量: {quality_score}/5)\n{reasoning_support}"
                )
                evidence_idx += 1
            # 如果没有推理支持，跳过这条证据（不再展示无用的状态语句）
        
        if not evidence_parts:
            return "（未检索到相关证据）"
        
        # 【验证日志】打印推理支持加载情况
        total_evidence = len(evidence_parts)
        print(f"[RAG格式化] 生成了 {total_evidence} 条证据，其中 {has_reasoning_support_count} 条包含【推理支持】")
        
        return "\n\n".join(evidence_parts)
    
    def retrieve_by_category_with_scores_batch(
        self, queries: List[str], top_k: int = 4
    ) -> List[Tuple[dict, List[float]]]:
        """
        【批量版】按类别检索多个 query 的证据，一次 BGE 编码 + 一次矩阵乘。
        
        Args:
            queries: 检索 query 列表
            top_k: 每个 query 检索的 top-k 数量（默认 4，对应 4 个类别）
            
        Returns:
            List[Tuple[dict, List[float]]]: 每个 query 的 (category_evidence, similarity_scores)
                - category_evidence 包含 quality_score 字段
        """
        if not queries:
            return []
        
        # 确保所有数据已加载
        self._load_embeddings()
        self._load_texts()
        self._load_reasoning_supports()
        self._load_reasoning_scores()
        self._load_index_mapping()
        
        # 【关键优化】批量编码所有 queries（一次 BGE forward）
        query_embeddings = self.encode_queries_batch(queries)
        
        # 批量计算相似度矩阵 (n_queries, n_embeddings)
        similarities = np.dot(query_embeddings, self.embeddings.T)
        
        category_names = ["Depression", "Anxiety", "Mix", "Others"]
        n_embeddings = similarities.shape[1]
        
        all_results = []
        for i in range(len(queries)):
            # 【优化】使用 argpartition 获取 top-k
            if top_k < n_embeddings:
                top_k_unsorted = np.argpartition(similarities[i], -top_k)[-top_k:]
                top_indices = top_k_unsorted[np.argsort(similarities[i, top_k_unsorted])[::-1]]
            else:
                top_indices = np.argsort(similarities[i])[::-1][:top_k]
            
            # 构建结果
            category_evidence = {}
            similarity_scores = []
            
            for j, category in enumerate(category_names):
                if j < len(top_indices):
                    idx = top_indices[j]
                    text = self.texts[idx]
                    score = float(similarities[i, idx])
                    reasoning_support = ""
                    if self.reasoning_supports and idx < len(self.reasoning_supports):
                        reasoning_support = self.reasoning_supports[idx] or ""
                    # 获取质量分数
                    quality_score = 0
                    if self.reasoning_scores and idx < len(self.reasoning_scores):
                        quality_score = self.reasoning_scores[idx] or 0
                    category_evidence[category] = {
                        "text": text,
                        "score": score,
                        "reasoning_support": reasoning_support,
                        "quality_score": quality_score
                    }
                    similarity_scores.append(score)
                else:
                    category_evidence[category] = None
            
            all_results.append((category_evidence, similarity_scores))
        
        return all_results


# 全局单例实例
_global_retriever = None


def get_rag_retriever(**kwargs) -> RAGRetriever:
    """获取全局 RAG 检索器实例"""
    global _global_retriever
    if _global_retriever is None:
        _global_retriever = RAGRetriever(**kwargs)
    return _global_retriever
