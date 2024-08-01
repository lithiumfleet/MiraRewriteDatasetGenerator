import json
from json import JSONDecodeError
import os
from tkinter import NO
from typing import TypeAlias
from dataclasses import dataclass
from openai import NotGiven, OpenAI
from sympy import N

# Options
class Opt:
    apikey = os.getenv("DSKEY")
    baseurl = "https://api.deepseek.com"
    dataset:list = []

class Msg:
    def __init__(self, role:str, content:str):
        self.role = role
        self.content = content

    def __repr__(self) -> str:
        return f"\n{self.role}: {self.content}\n"

    def to_dict(self) -> dict[str:str]:
        return {
            "role": self.role,
            "content": self.content
        }

Messages: TypeAlias = list[Msg]

def create_client() -> OpenAI:
    return OpenAI(api_key=Opt.apikey, base_url=Opt.baseurl)

def chat(client:OpenAI, question:str, history:Messages|None=None, temperature=0.0, sys_prompt:str|None=None, json_path:list[str]|None=None) -> tuple[str, Messages]:
    if history is None:
        history = list()
    if sys_prompt is not None:
        history.append({"role":"system", "content":sys_prompt})
    history.append({"role":"user", "content":question})
    use_json = json_path is not None and len(json_path) > 0
    if use_json:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=history,
            temperature=temperature,
            response_format={ 'type': 'json_object' }
        )
    else:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=history,
            temperature=temperature
        )
    resp = response.choices[0].message.content
    if len(resp.strip().replace(" ", "")) < 1:
        raise RuntimeError("Error: the model returns empty.")
    history.append({"role":"assistant", "content":resp})

    if use_json:
        try:
            resp = json.loads(resp)
        except JSONDecodeError as err:
            print(err)
            print(f"Error: Cannot decode {resp}")
            raise RuntimeError("Error: json decode error.")

        try:
            for path in json_path:
                resp = resp[path]
        except KeyError as err:
            print(err)
            print(f"Error: model did not follow the json prompt, get no key {path} in {resp}")
            raise RuntimeError("Error: model output error.")

    return resp, history
