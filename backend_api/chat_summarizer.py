from anthropic import AsyncAnthropic
import asyncio
import os
import base64
import sys

### This script first reads a research paper, summarize the key experiement, then check if the summarzied key experiments have been repeated or conducted in the second paper

def main():
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    param1 = sys.argv[1] #This is the target paper we are going to validate
    param2 = sys.argv[2] #This is the paper we would like use to validate

    target_paper = read_pdf_file(param1)
    paper_use_for_check = read_pdf_file(param2)

    summarize_prompt = "Analyze the research paper and summarize the key experiments. The summarization should include detailed experiment condition, and experiment result"

    summary = asyncio.run(analyze(client, target_paper, summarize_prompt))

    validation_prompt = "Please validate the following key experiment has been conducted or validated in the uploaded research paper:" + summary

    validation = asyncio.run(analyze(client, paper_use_for_check, validation_prompt))
     

#Read pdf file from given file path
def read_pdf_file(file_path):
    with open(file_path,'rb') as in_file:
        pdf_content = in_file.read()
    return pdf_content

#This method calls the arthropic API to analyze a research paper
async def analyze(client, paper, prompt):
    response = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=8192,
        temperature=0,
        messages=[
            {"role": "user", "content": [{
                                            "type": "document",
                                            "source":  {
                                                "type": "base64",
                                                "media_type": "application/pdf",
                                                "data": base64.b64encode(paper).decode()
                                            }},
                                          {
                                            "type": "text",
                                            "text": prompt
                                          }]}
        ],
        stream=True
    )
    response_str = ""
    async for message in response:
        if message.type =="content_block_delta":
            print(message.delta.text, end="", flush=True)
            response_str += message.delta.text
    return response_str        

if __name__ == "__main__":
    main()