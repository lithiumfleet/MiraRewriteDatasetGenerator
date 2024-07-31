from decimal import Clamped
from functools import cache
import select
import sqlite3
from typing import Any, Protocol
import random

class DatabaseLike(Protocol):
    def random_get(self, num:int, seed:int) -> Any: ...
    def close(self): ...
    @property 
    @cache
    def size(self) -> int: ...

class SQliteDB:
    def __init__(self, db_path:str) -> None:
        self.conn = sqlite3.connect(db_path)
        self.table_name = "train"

    def random_get(self, num:int, seed:int) -> Any:
        random.seed(seed)
        rand_offset = random.sample(range(self.size), num)

        random_rows = []
        cur = self.conn.cursor()
        for offset in rand_offset:
            cur.execute(f"SELECT * FROM {self.table_name} LIMIT 1 OFFSET {offset}")
            random_rows.append(cur.fetchone())
        return random_rows

    def close(self):
        self.conn.close()

    @property    
    @cache
    def size(self) -> int:
        cur = self.conn.cursor()
        res = cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        return res.fetchone()[0]

    @classmethod
    def connect_db(cls, db_path:str) -> 'SQliteDB':
        return cls(db_path)