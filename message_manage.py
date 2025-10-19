import threading
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class MessageManager:
    def __init__(self, max_history: int = 10):
        """
        初始化消息管理器

        :param max_history: 每个 session 最多保留的历史消息数量（不包括 system 消息）
        """
        self.max_history = max_history
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()  # 保证线程安全

    def set_system_message(self, session_id: str, content: str) -> None:
        """为指定 session 设置 system 消息（会覆盖旧的）"""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {
                    "system": None,
                    "history": []  # 只存 user/assistant 对话
                }
            self._sessions[session_id]["system"] = Message(role="system", content=content)

    def add_user_message(self, session_id: str, content: str) -> None:
        """添加用户消息"""
        self._add_message(session_id, "user", content)

    def add_assistant_message(self, session_id: str, content: str) -> None:
        """添加助手回复"""
        self._add_message(session_id, "assistant", content)

    def _add_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {
                    "system": None,
                    "history": []
                }
            history = self._sessions[session_id]["history"]
            history.append(Message(role=role, content=content))
            # 限制历史长度（只保留最近的 max_history 条 user/assistant 消息）
            if len(history) > self.max_history * 2:  # 每轮对话含 user + assistant
                # 保留最后 max_history * 2 条
                self._sessions[session_id]["history"] = history[-(self.max_history * 2):]

    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        """获取可用于 Ollama /api/chat 的 messages 列表"""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return []

            messages = []
            # 添加 system 消息（如果有）
            if session["system"]:
                messages.append(session["system"].to_dict())
            # 添加历史对话
            for msg in session["history"]:
                messages.append(msg.to_dict())
            return messages

    def clear_session(self, session_id: str) -> None:
        """清除指定 session 的所有消息"""
        with self._lock:
            self._sessions.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        """列出所有 session ID"""
        with self._lock:
            return list(self._sessions.keys())

    def delete_session(self, session_id: str) -> bool:
        """删除 session，返回是否删除成功"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False