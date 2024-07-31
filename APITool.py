import os
from typing import TypeAlias
from dataclasses import dataclass
from openai import OpenAI

# Options
class Opt:
    apikey = os.getenv("DSKEY")
    baseurl = "https://api.deepseek.com/v1"
    dataset:list = []

class Msg:
    def __init__(self, role:str, content:str):
        self.role = role
        self.content = content

    def __repr__(self) -> str:
        return f"\n{self.role}: {self.content}\n"

Messages: TypeAlias = list[Msg]

def create_client() -> OpenAI:
    return OpenAI(api_key=Opt.apikey, base_url=Opt.baseurl)

def chat(client:OpenAI, question:str, history:Messages=[], temperature=0.0, sys_prompt:str|None=None) -> tuple[str, Messages]:
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
