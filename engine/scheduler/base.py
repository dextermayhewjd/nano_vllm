# engine/scheduler/base.py
from abc import ABC, abstractmethod
from typing import Optional
from engine.request import Request

class Scheduler(ABC):
    """最小抽象：调度一批 Request。"""

    @abstractmethod
    def add_request(self, req: Request) -> None:
        """把一个新的请求放进调度队列。"""
        raise NotImplementedError

    @abstractmethod
    def has_next(self) -> bool:
        """队列里还有待执行的请求吗？"""
        raise NotImplementedError

    @abstractmethod
    def get_next(self) -> Optional[Request]:
        """取出下一个要执行的请求。"""
        raise NotImplementedError