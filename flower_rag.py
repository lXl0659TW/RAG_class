import os, json, ollama, re, time
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

class FlowerShopRAG:
    def __init__(self, model_name="llama3.2-vision:11b-instruct-fp16", 
                 img_db_dir="img_db", 
                 db_file="flower_db.json"):
        self.model_name = model_name
        self.img_db_dir = img_db_dir
        self.db_file = db_file
        
        # CPU 環境專用，明確指定本機連接埠，避免 Flask 找不到 Ollama
        print(f"\n[DEBUG] 🔧 正在初始化 Ollama 客戶端連線至 http://127.0.0.1:11434 ...")
        self.client = ollama.Client(host='http://127.0.0.1:11434', timeout=None)
        
        print(f"[DEBUG] 📂 正在確認產品資料庫狀態...")
        self.database = self._load_or_build_index()
        print(f"[DEBUG] ✅ FlowerShopRAG 初始化完成！\n" + "="*50)

    def _load_or_build_index(self):
        if os.path.exists(self.db_file):
            with open(self.db_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"📦 已載入現有資料庫 ({len(data)} 件產品)。")
                return data

        if not os.path.exists(self.img_db_dir):
            print(f"⚠️ 找不到 {self.img_db_dir} 資料夾。")
            return []

        image_files = sorted([f for f in os.listdir(self.img_db_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"🚀 [CPU 模式] 正在對 {len(image_files)} 張圖片建立索引...")
        print(f"💡 使用模型: {self.model_name} (請耐心等候，CPU 推理較慢)")

        flower_db = []
        total_start = time.perf_counter()

        pbar = tqdm(image_files, desc="分析進度", unit="張")
        
        for filename in pbar:
            path = os.path.join(self.img_db_dir, filename)
            pbar.set_postfix_str(f"正在處理: {filename}")
            single_start = time.perf_counter()
            
            prompt = """請分析這張照片並以 JSON 格式回傳 (繁體中文)：
            {
              "flower_type": "種類", "main_color": "主色調", "petal_shape": "花瓣細節",
              "center_detail": "花心描述", "visual_style": "特徵風格"
            }"""

            try:
                response = self.client.chat(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': prompt, 'images': [path]}],
                    format='json',
                    options={"temperature": 0, "num_thread": 8}
                )
                
                item_data = json.loads(response['message']['content'])
                item_data['id'] = filename
                flower_db.append(item_data)
                
                single_elapsed = time.perf_counter() - single_start
                tqdm.write(f"✅ {filename} 處理完成 (耗時: {single_elapsed:.1f} 秒)")
                
            except Exception as e:
                tqdm.write(f"❌ {filename} 失敗: {str(e)[:50]}...")

        total_elapsed = time.perf_counter() - total_start
        print(f"\n✨ 索引建立完畢！總耗時: {total_elapsed/60:.2f} 分鐘")

        with open(self.db_file, "w", encoding="utf-8") as f:
            json.dump(flower_db, f, ensure_ascii=False, indent=4)
        return flower_db

    def get_recommendation(self, user_image_path):
        if not self.database: return "❌ 資料庫空。"

        print(f"🔍 正在分析客戶照片 (CPU 推理中)...")
        start_time = time.perf_counter()

        try:
            user_res = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': "請詳細分析這張照片中花卉的種類、顏色、形狀與風格特徵，用繁體中文回報。", 
                           'images': [user_image_path]}],
                options={"num_thread": 8}
            )
            user_analysis = user_res['message']['content']
            
            final_prompt = f"""
            你現在是『山洞口花店』的專業店員。
            
            【店內產品資料庫】：
            {json.dumps(self.database, ensure_ascii=False)}

            【客戶照片特徵描述】：
            "{user_analysis}"

            【任務指令】：
            請比對客戶照片與資料庫，找出最相似的 2 個產品。
            
            【回傳規範（極其重要）】：
            1. 必須直接使用資料庫中的 "id" (例如: lilly1.jpg) 來代表產品，禁止自創名稱。
            2. 格式請嚴格遵守：
               推薦 1：[請填入 id]
               相似理由：(請簡短說明顏色或形狀的匹配點)
               推薦 2：[請填入 id]
               相似理由：(請簡短說明風格或結構的匹配點)
            3. 最後，請用溫馨的口吻給予一段 50 字以內的繁體中文建議。
            """

            res = self.client.chat(
                model=self.model_name, 
                messages=[{'role': 'user', 'content': final_prompt}],
                options={"num_thread": 8}
            )
            
            full_response = res['message']['content']
            elapsed = time.perf_counter() - start_time
            print(f"✨ 服務完成！本次耗時: {elapsed:.2f} 秒")
            
            valid_ids = [item['id'] for item in self.database]
            recommend_files = []
            for word in re.split(r'[\s，：；\[\]]+', full_response):
                if word in valid_ids and word not in recommend_files:
                    recommend_files.append(word)
            
            recommend_files = recommend_files[:2]
            
            if recommend_files:
                self._display(recommend_files)
            else:
                print("⚠️ 警告：模型回傳內容未包含有效的產品 ID。")
                
            return full_response
        except Exception as e:
            return f"❌ 錯誤: {e}"

    def _display(self, filenames):
        valid_paths = [os.path.join(self.img_db_dir, f) for f in filenames if os.path.exists(os.path.join(self.img_db_dir, f))]
        if not valid_paths: return
        plt.figure(figsize=(8, 4))
        for i, p in enumerate(valid_paths):
            plt.subplot(1, len(valid_paths), i + 1)
            plt.imshow(Image.open(p))
            plt.title(os.path.basename(p))
            plt.axis('off')
        plt.show()
    
    # 這是給 Flask 網頁使用的核心函數（已加入豐富的 DEBUG 訊息）
    def get_rec_data(self, user_image_path):
        print(f"\n" + "="*50)
        print(f"[DEBUG] 📩 接收到前端上傳圖片要求。")
        print(f"[DEBUG] 📍 檔案存放路徑: {user_image_path}")

        if not self.database: 
            print(f"[DEBUG] ❌ 錯誤：資料庫為空，無法進行推薦。")
            return {"error": "資料庫空。"}

        try:
            # 1. 讀取並縮放圖片
            print(f"[DEBUG] 📸 正在讀取並壓縮使用者圖片以提升 CPU 推理速度...")
            start_time = time.perf_counter()
            
            with Image.open(user_image_path) as img:
                print(f"[DEBUG] 📖 成功讀取原始檔案 (尺寸: {img.size}, 格式: {img.format})")
                img.thumbnail((512, 512)) # 縮放至 512px 以內
                temp_path = user_image_path + "_temp.jpg"
                img.convert("RGB").save(temp_path, "JPEG")
                print(f"[DEBUG] 💾 縮減圖已暫存至: {temp_path} (新尺寸: {img.size})")

            # 2. 呼叫 Ollama 分析客戶照片
            print(f"[DEBUG] 🧠 [步驟 1/2] 正在傳送暫存圖至 Ollama ({self.model_name}) 分析花卉特徵...")
            ai_start = time.perf_counter()
            
            user_res = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': "請詳細分析這張花的顏色、花瓣形狀與整體風格，用繁體中文回答。", 
                           'images': [temp_path]}], # 已修正：改用暫存的縮小圖
                options={"num_thread": 8, "temperature": 0}
            )
            user_analysis = user_res['message']['content']
            ai_elapsed = time.perf_counter() - ai_start
            print(f"[DEBUG] ✨ [步驟 1/2] 分析完成！耗時: {ai_elapsed:.2f} 秒")
            print(f"[DEBUG] 📝 客戶照片分析結果片段:\n{user_analysis[:150]}...\n")
            
            # 3. 呼叫 Ollama 比對 RAG 資料庫
            print(f"[DEBUG] 🧠 [步驟 2/2] 正在進行 RAG 資料庫比對與產品推薦...")
            rag_start = time.perf_counter()
            
            final_prompt = f"""
            你現在是『山洞口花店』的專業資深店員。
            
            【店內產品資料庫】：
            {json.dumps(self.database, ensure_ascii=False)}

            【客戶照片特徵描述】：
            "{user_analysis}"

            【任務指令】：
            1. 即使特徵沒有 100% 吻合，也必須從資料庫中挑選出最『視覺接近』的 2 個產品。
            2. 嚴禁回答「找不到」或「無法推薦」。
            3. 必須直接回傳產品的 "id" (例如: lilly1.jpg)。
            
            【回傳規範】：
            推薦 1：[id]
            相似理由：(說明顏色或形狀的相似點)
            推薦 2：[id]
            相似理由：(說明風格的相似點)
            最後請給予一段 50 字內溫馨的建議。
            """

            res = self.client.chat(
                model=self.model_name, 
                messages=[{'role': 'user', 'content': final_prompt}],
                options={"num_thread": 8, "temperature": 0.2}
            )
            
            full_response = res['message']['content']
            rag_elapsed = time.perf_counter() - rag_start
            print(f"[DEBUG] ✨ [步驟 2/2] RAG 推薦完成！耗時: {rag_elapsed:.2f} 秒")
            
            # 4. 提取有效的 ID
            print(f"[DEBUG] 🔍 正在從 AI 回傳文字中解析產品 ID...")
            valid_ids = [item['id'] for item in self.database]
            recommend_files = []
            for word in re.split(r'[\s，：；\[\]\n]+', full_response):
                clean_word = word.strip()
                if clean_word in valid_ids and clean_word not in recommend_files:
                    recommend_files.append(clean_word)
            
            print(f"[DEBUG] 🎯 解析成功！成功比對到的推薦 ID: {recommend_files[:2]}")

            # 5. 清理臨時圖
            if os.path.exists(temp_path): 
                os.remove(temp_path)
                print(f"[DEBUG] 🧹 暫存圖已安全刪除。")

            total_elapsed = time.perf_counter() - start_time
            print(f"[DEBUG] 🏆 網頁推薦功能執行完畢！總耗時: {total_elapsed:.2f} 秒")
            print("="*50 + "\n")

            return {
                "user_analysis": user_analysis,
                "recommend_ids": recommend_files[:2],
                "ai_message": full_response
            }
        except Exception as e:
            print(f"[DEBUG] ❌ 發生例外錯誤: {str(e)}")
            # 確保發生錯誤時也會清理暫存檔
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
            return {"error": str(e)}
