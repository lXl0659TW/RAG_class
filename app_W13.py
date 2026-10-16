from flask import Flask, render_template, request, send_from_directory
from flower_rag import FlowerShopRAG
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 初始化 Class (這會載入資料庫)
assistant = FlowerShopRAG(model_name="qwen2.5vl-cpu:latest")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # 儲存上傳的照片
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            
            # 呼叫 AI 獲取結果
            result = assistant.get_rec_data(path)
            result['user_img_url'] = f'uploads/{file.filename}'

    return render_template('index.html', result=result)

# 讓網頁能讀取 img_db 內的圖片
@app.route('/img_db/<path:filename>')
def serve_db_img(filename):
    return send_from_directory('img_db', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8090, threaded=True)
