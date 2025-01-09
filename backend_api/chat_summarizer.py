from anthropic import AsyncAnthropic
import time
import asyncio
import os
import base64

system_prompt_template = """You are {name} a virtual assistant created by {owner}. Today is {date}. You provide responses to questions that are clear, straightforward, and factually accurate, without speculation or falsehood. Given the following context, please answer each question truthfully to the best of your abilities based on the provided information. Answer each question with a brief summary followed by several bullet points. 

Example:
Summary of answer
- bullet point 1
- bullet point 2
...

<context>
{context}
</context>
"""

with open("elife-26403.pdf",'rb') as in_file:
    context_content = in_file.read()

system_prompt = system_prompt_template.format(
    name="Research Paper Analyzer",
    owner="Ying",
    date="Dec. 20th, 2024",
    context=context_content
)

async def chat_func():
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=8192,
        temperature=0,
      #  system=system_prompt,
        messages=[
            {"role": "user", "content": [{
                                            "type": "document",
                                            "source":  {
                                                "type": "base64",
                                                "media_type": "application/pdf",
                                                "data": base64.b64encode(context_content).decode()
                                            }},
                                          {
                                            "type": "text",
                                            "text": "Analyze the research paper and summarize the key experiments. The summarization would include detailed experiment condition, and experiment result"  
                                          }]}
        ],
        stream=True
    )
    async for message in response:
        if message.type =="content_block_delta":
            print(message.delta.text, end="", flush=True)

   # print(result.content)
    # async for message in result:
    #     print(message.content, flush=True, end="")


asyncio.run(chat_func())