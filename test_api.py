import requests
import json
import uuid

def test_chat_api():
    """
    测试 /chat 接口
    """
    url = "http://127.0.0.1:8000/chat"
    
    # 构造请求数据
    payload = {
        "query": "你能做什么",
        "thread_id": str(uuid.uuid4())  # 生成一个新的会话 ID
    }
    
    headers = {
        "Content-Type": "application/json"
    }

    print(f"Sending request to {url}...")
    print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        response = requests.post(url, json=payload, headers=headers)
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\nResponse:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # 简单的断言
            assert "response" in result
            assert "intent" in result
            assert result["thread_id"] == payload["thread_id"]
            print("\n✅ Test Passed!")
        else:
            print(f"\n❌ Request failed: {response.text}")

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")

if __name__ == "__main__":
    test_chat_api()
