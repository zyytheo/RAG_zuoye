import requests


def test_ask_endpoint(question):
    url = "http://localhost:8000/ask"
    params = {"question": question}

    with requests.get(url, params=params, stream=True) as response:
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=128):
                if chunk: 
                    print(chunk.decode('utf-8'), end='', flush=True) 
        else:
            print(f"error: {response.status_code} - {response.text}")


if __name__ == "__main__":
    def build_prompt(user_q: str) -> str:
        return (
            "Please answer strictly based on your local knowledge base;"
            "If the context is empty or insufficient, just reply: No relevant information was found in the knowledge base."
            "Answer in concise English and do not reveal hints or internal thoughts."
            f"question：{user_q}"
        )

    test_question = build_prompt("Who are the executives of Xiaomi Mobile?")
    print(f"send question: {test_question}")
    test_ask_endpoint(test_question)