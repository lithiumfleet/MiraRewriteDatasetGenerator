import os
from openai import OpenAI

# Options
class Opt:
    apikey = os.getenv("DSKEY")
    baseurl = "https://api.deepseek.com/v1"
    dataset:list = []

class Msg:
    role: str
    content: str

def create_client() -> OpenAI:
    return OpenAI(api_key=Opt.apikey, base_url=Opt.baseurl)

def chat(client:OpenAI, question:str, history:list[Msg]=[], temperature=0.0, sys_prompt:str|None=None) -> tuple[str, list]:
    if sys_prompt is not None:
        history.append({"role":"system", "content":sys_prompt})
    history.append({"role":"user", "content":question})
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=history,
        stream=False,
        temperature=temperature
    )
    resp = response.choices[0].message.content
    history.append({"role":"assistant", "content":resp})
    return resp, history
