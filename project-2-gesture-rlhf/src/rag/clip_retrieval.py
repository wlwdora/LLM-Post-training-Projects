import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.preprocessing import normalize


class GestureRAGRetriever:
    """
    基于 CLIP 的多模态向量检索模块
    功能：以图搜图、以文搜图，为手势识别系统提供 RAG 兜底检索
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[RAG] 加载 CLIP 模型: {model_name} → {self.device}")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # 向量库
        self.image_embeddings = None   # [N, D]
        self.text_embeddings = None    # [N, D]
        self.metadata = []             # 每条记录的原信息
        self.dim = self.model.config.projection_dim

    @torch.no_grad()
    def encode_image(self, image_input):
        """编码单张或多张图片"""
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("image_input 必须是路径或 PIL.Image")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)
        # L2 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

    @torch.no_grad()
    def encode_text(self, text):
        """编码文本"""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()

    def build_index(self, image_paths, texts, metadatas=None):
        """
        构建向量索引库
        :param image_paths: 图片路径列表
        :param texts: 对应文本描述列表（如"合影剪刀手"）
        :param metadatas: 额外信息列表（如场景标签、标准回答）
        """
        print(f"[RAG] 构建索引库，共 {len(image_paths)} 条记录...")

        batch_size = 32
        all_img_emb = []
        all_txt_emb = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]

            # 编码图片
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            img_feat = self.model.get_image_features(**inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            # 编码文本
            txt_inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True).to(self.device)
            txt_feat = self.model.get_text_features(**txt_inputs)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            all_img_emb.append(img_feat.cpu().numpy())
            all_txt_emb.append(txt_feat.cpu().numpy())

        self.image_embeddings = np.vstack(all_img_emb).astype("float32")
        self.text_embeddings = np.vstack(all_txt_emb).astype("float32")

        # 多模态融合：图片向量 + 文本向量（平均），也可加权
        fused = (self.image_embeddings + self.text_embeddings) / 2
        self.fused_embeddings = normalize(fused, norm='l2')

        self.metadata = metadatas if metadatas else [{"path": p, "text": t} for p, t in zip(image_paths, texts)]
        print(f"[RAG] 索引构建完成，维度: {self.fused_embeddings.shape}")

    def search_by_image(self, query_image, top_k=3):
        """以图搜图：输入手势图片，检索最相似的历史样本"""
        query_emb = self.encode_image(query_image)
        # 与融合向量比对
        similarities = (self.fused_embeddings @ query_emb.T).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "score": float(similarities[idx]),
                "metadata": self.metadata[idx]
            })
        return results

    def search_by_text(self, query_text, top_k=3):
        """以文搜图：输入描述，检索对应手势"""
        query_emb = self.encode_text(query_text)
        similarities = (self.fused_embeddings @ query_emb.T).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "score": float(similarities[idx]),
                "metadata": self.metadata[idx]
            })
        return results

    def save_index(self, save_dir="./rag_index"):
        """保存向量库"""
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "fused_embeddings.npy"), self.fused_embeddings)
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print(f"[RAG] 索引已保存至 {save_dir}")

    def load_index(self, save_dir="./rag_index"):
        """加载向量库"""
        self.fused_embeddings = np.load(os.path.join(save_dir, "fused_embeddings.npy"))
        with open(os.path.join(save_dir, "metadata.json"), "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"[RAG] 索引已加载，共 {len(self.metadata)} 条记录")


def build_gesture_rag_index(image_dir, metadata_json, output_dir="./rag_index"):
    """
    从手势数据集构建 RAG 索引
    :param image_dir: 手势图片目录
    :param metadata_json: 包含 {图片名: 描述} 的 JSON
    """
    with open(metadata_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    image_paths = []
    texts = []
    metadatas = []

    for img_name, info in meta.items():
        path = os.path.join(image_dir, img_name)
        if os.path.exists(path):
            image_paths.append(path)
            texts.append(info["description"])
            metadatas.append({
                "image_path": path,
                "description": info["description"],
                "scene": info.get("scene", "unknown"),
                "gesture_type": info.get("gesture_type", "unknown")
            })

    retriever = GestureRAGRetriever()
    retriever.build_index(image_paths, texts, metadatas)
    retriever.save_index(output_dir)
    return retriever


if __name__ == "__main__":
    # 示例：构建索引并测试检索
    retriever = GestureRAGRetriever()

    # 假设已有图片和描述
    demo_images = ["./demo/party_01.jpg", "./demo/game_01.jpg"]
    demo_texts = ["朋友聚会自拍剪刀手", "石头剪刀布游戏出招"]

    retriever.build_index(demo_images, demo_texts)

    # 以图搜图
    results = retriever.search_by_image("./demo/party_02.jpg", top_k=2)
    print("\n以图搜图结果:")
    for r in results:
        print(f"  相似度: {r['score']:.3f} | 描述: {r['metadata']['text']}")

    # 保存
    retriever.save_index("./output/rag_index")