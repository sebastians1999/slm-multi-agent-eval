from langchain_openai import ChatOpenAI
import openai


if __name__ == "__main__":
   
    print("\n=== Using OpenAI Client ===")
    client = openai.OpenAI(
        api_key="EMPTY",
        base_url="https://sebastian-schmuelling--slm-server-vllmserver-serve-dev.modal.run/v1/"
    )

    response = client.chat.completions.create(
        #model="microsoft/Phi-3.5-mini-instruct",
        model='microsoft/Phi-3.5-mini-instruct',
        messages=[{"role": "user", "content": "Hello!"}],
        temperature=0,
    )

    print("Content:", response.choices[0].message.content)

    
    response_dict = response.model_dump()
    print("Energy consumption:", response_dict.get('energy_consumption', 'Not found'))
