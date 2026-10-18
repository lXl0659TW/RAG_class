import os
import chromadb
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

# 配置參數
CHROMA_PATH = "./chroma_flower_db"
IMG_DB_DIR = "img_db"

# 初始化 ChromaDB Client (唯讀)
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name="flowers")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # 接收用戶文字輸入
        user_flower = request.form.get('flower_type', '').strip()
        user_desc = request.form.get('description', '').strip()
        
        # 組合 Query
        query_text = f"{user_flower} {user_desc}"
        
        # --- Hybrid RAG 檢索策略 ---
        # 1. 優先嘗試「品種過濾」(Hard Filter)
        search_results = collection.query(
            query_texts=[query_text],
            n_results=2,
            where={"flower_type": user_flower} if user_flower else None
        )

        # 2. Fallback：如果該品種沒貨，改為「全店相似度檢索」
        is_fallback = False
        if not search_results['ids'] or len(search_results['ids'][0]) == 0:
            is_fallback = True
            search_results = collection.query(
                query_texts=[query_text],
                n_results=2
            )

        # 3. 封裝結果 (類比原 app.py 的 result 結構)
        recommendations = []
        for i in range(len(search_results['ids'][0])):
            recommendations.append({
                "filename": search_results['ids'][0][i],
                "description": search_results['documents'][0][i],
                "distance": round(search_results['distances'][0][i], 4)
            })

        result = {
            "user_query": query_text,
            "is_fallback": is_fallback,
            "recommendations": recommendations
        }

    return render_template('index.html', result=result)

@app.route('/img_db/<path:filename>')
def serve_db_img(filename):
    return send_from_directory(IMG_DB_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8091, threaded=True)