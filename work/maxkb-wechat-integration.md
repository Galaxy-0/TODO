# MaxKB 微信公众号集成完整方案

## 答案：可以接入！

MaxKB 提供标准 REST API，完全支持接入微信公众号。实现方式是开发一个中间服务层，连接微信公众号和 MaxKB。

## 集成架构

```
微信用户 ←→ 微信公众号 ←→ 中间服务器 ←→ MaxKB API ←→ 知识库
```

## 详细实现步骤

### 1. 部署 MaxKB（1天）

```bash
# 使用 Docker Compose 部署
wget https://github.com/1Panel-dev/MaxKB/releases/download/v1.4.0/docker-compose.yml
docker-compose up -d

# 访问 http://localhost:8080
# 默认账号：admin / MaxKB@123..
```

### 2. 配置知识库（1天）

1. 登录 MaxKB 管理界面
2. 创建知识库，上传政府题库文档
3. 创建应用，获取 API Key
4. 测试问答效果

### 3. 开发微信公众号中间服务（2-3天）

完整代码示例：

```python
# app.py - Flask 服务
from flask import Flask, request, make_response
import xml.etree.ElementTree as ET
import requests
import hashlib
import time

app = Flask(__name__)

# 配置
WECHAT_TOKEN = "your_wechat_token"
MAXKB_API_KEY = "your_maxkb_api_key"
MAXKB_API_URL = "http://localhost:8080/api/v1"

# 微信验证
@app.route('/wechat', methods=['GET'])
def wechat_verify():
    signature = request.args.get('signature')
    timestamp = request.args.get('timestamp')
    nonce = request.args.get('nonce')
    echostr = request.args.get('echostr')
    
    # 验证签名
    if check_signature(signature, timestamp, nonce):
        return echostr
    return "Invalid"

# 处理微信消息
@app.route('/wechat', methods=['POST'])
def wechat_handler():
    # 解析 XML 消息
    xml_data = request.data
    root = ET.fromstring(xml_data)
    
    msg_type = root.find('MsgType').text
    from_user = root.find('FromUserName').text
    to_user = root.find('ToUserName').text
    
    if msg_type == 'text':
        content = root.find('Content').text
        
        # 调用 MaxKB API
        answer = query_maxkb(content, from_user)
        
        # 构造回复
        reply = create_text_reply(from_user, to_user, answer)
        
        response = make_response(reply)
        response.content_type = 'application/xml'
        return response
    
    return "success"

# 调用 MaxKB API
def query_maxkb(question, user_id):
    headers = {
        'Authorization': f'Bearer {MAXKB_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'question': question,
        'user_id': user_id,
        'stream': False
    }
    
    try:
        response = requests.post(
            f'{MAXKB_API_URL}/chat',
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('answer', '抱歉，我暂时无法回答这个问题。')
        else:
            return '系统繁忙，请稍后再试。'
    except Exception as e:
        print(f"MaxKB API 错误: {e}")
        return '系统异常，请联系管理员。'

# 创建文本回复
def create_text_reply(to_user, from_user, content):
    reply = f"""
    <xml>
        <ToUserName><![CDATA[{to_user}]]></ToUserName>
        <FromUserName><![CDATA[{from_user}]]></FromUserName>
        <CreateTime>{int(time.time())}</CreateTime>
        <MsgType><![CDATA[text]]></MsgType>
        <Content><![CDATA[{content}]]></Content>
    </xml>
    """
    return reply.strip()

# 验证微信签名
def check_signature(signature, timestamp, nonce):
    tmp_list = [WECHAT_TOKEN, timestamp, nonce]
    tmp_list.sort()
    tmp_str = ''.join(tmp_list)
    tmp_str = hashlib.sha1(tmp_str.encode()).hexdigest()
    return tmp_str == signature

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

### 4. 部署中间服务（0.5天）

```bash
# 安装依赖
pip install flask requests gunicorn

# 使用 Gunicorn 运行
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# 或使用 PM2（Node.js）
pm2 start app.py --interpreter python3
```

### 5. 配置微信公众号（0.5天）

1. 登录微信公众平台
2. 进入"设置与开发" → "基本配置"
3. 配置服务器地址：`https://your-domain.com/wechat`
4. 设置 Token（与代码中一致）
5. 选择消息加密方式（建议明文模式测试）

### 6. 进阶功能（可选）

```python
# 1. 会话上下文管理
session_context = {}

def query_maxkb_with_context(question, user_id):
    # 获取用户历史对话
    context = session_context.get(user_id, [])
    
    # 构建带上下文的请求
    data = {
        'question': question,
        'user_id': user_id,
        'conversation_id': get_or_create_conversation(user_id),
        'stream': False
    }
    
    # 更新会话历史
    context.append({'role': 'user', 'content': question})
    session_context[user_id] = context[-10:]  # 保留最近10轮

# 2. 多轮对话支持
def handle_multi_turn_conversation(user_id, message):
    # 检查是否在对话流程中
    if is_in_conversation_flow(user_id):
        return continue_conversation(user_id, message)
    else:
        return start_new_conversation(user_id, message)

# 3. 图文消息支持
def create_news_reply(to_user, from_user, articles):
    items = ''.join([f"""
    <item>
        <Title><![CDATA[{article['title']}]]></Title>
        <Description><![CDATA[{article['description']}]]></Description>
        <PicUrl><![CDATA[{article['pic_url']}]]></PicUrl>
        <Url><![CDATA[{article['url']}]]></Url>
    </item>
    """ for article in articles])
    
    return f"""
    <xml>
        <ToUserName><![CDATA[{to_user}]]></ToUserName>
        <FromUserName><![CDATA[{from_user}]]></FromUserName>
        <CreateTime>{int(time.time())}</CreateTime>
        <MsgType><![CDATA[news]]></MsgType>
        <ArticleCount>{len(articles)}</ArticleCount>
        <Articles>{items}</Articles>
    </xml>
    """
```

## 运维监控

### 1. 日志配置

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='wechat_maxkb.log'
)

# 记录关键操作
logger = logging.getLogger(__name__)
logger.info(f"用户 {user_id} 提问: {question}")
logger.info(f"MaxKB 回答: {answer}")
```

### 2. 性能优化

```python
# Redis 缓存高频问题
import redis

r = redis.Redis(host='localhost', port=6379, db=0)

def query_with_cache(question):
    # 检查缓存
    cached = r.get(f"q:{question}")
    if cached:
        return cached.decode()
    
    # 查询 MaxKB
    answer = query_maxkb(question)
    
    # 缓存结果（1小时）
    r.setex(f"q:{question}", 3600, answer)
    
    return answer
```

## 总工作量

| 任务 | 工作量 |
|------|--------|
| MaxKB 部署配置 | 1天 |
| 知识库导入测试 | 1天 |
| 中间服务开发 | 2-3天 |
| 部署和联调 | 1天 |
| **总计** | **5-6天** |

## 注意事项

1. **消息限制**：微信文本消息最长 2048 字符，需要处理长回复
2. **响应时间**：微信要求 5 秒内响应，考虑异步处理
3. **并发处理**：使用 Gunicorn 多进程或 Celery 异步任务
4. **安全防护**：验证消息来源，防止恶意调用

## 优势总结

✅ **技术成熟**：MaxKB 提供标准 API，集成简单
✅ **功能完整**：支持多轮对话、上下文管理
✅ **易于维护**：通过 MaxKB 界面管理知识库
✅ **成本可控**：本地部署，无额外 API 费用

这个方案完全可行，而且已经有很多企业成功实施！