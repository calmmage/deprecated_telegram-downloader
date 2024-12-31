import json
from datetime import datetime, timedelta

from telethon.types import Channel, Chat, User


class ChatData:
    entity_type_map = {
        "Channel": Channel,
        "User": User,
        "Chat": Chat,
    }

    def __init__(self, entity, last_message_date):
        self.entity = entity
        if isinstance(last_message_date, str):
            last_message_date = datetime.fromisoformat(last_message_date)
        self.last_message_date = last_message_date
        self.finished_downloading = False

    def to_json(self):
        data = {
            "entity": self.entity.to_json(),
            "last_message_date": self.last_message_date.isoformat(),
            "finished_downloading": self.finished_downloading,
        }
        return data

    @classmethod
    def from_json(cls, data):
        entity_data = json.loads(data["entity"])
        entity_type = entity_data.pop("_")
        entity = cls.load_entity(entity_type, entity_data)
        last_message_date = datetime.fromisoformat(data["last_message_date"])
        return cls(
            entity=entity,
            last_message_date=last_message_date,
            finished_downloading=data["finished_downloading"],
        )

    @classmethod
    def load_entity(cls, entity_type, entity_data):
        entity_class = cls.entity_type_map[entity_type]
        return entity_class(**entity_data)

    @property
    def entity_category(self):
        if isinstance(self.entity, User):
            return "bot" if getattr(self.entity, "bot", False) else "private chat"
        elif isinstance(self.entity, Chat):
            return "group"
        elif isinstance(self.entity, Channel):
            return "channel" if not self.entity.megagroup else "group"
        return "unknown"

    recent_threshold = timedelta(days=30)
    big_threshold = 1000

    def get_is_recent(self, recent_threshold: timedelta = timedelta(days=30)):
        if self.last_message_date is None:
            return False
        current_time = datetime.now(self.last_message_date.tzinfo)
        return (current_time - self.last_message_date) < recent_threshold

    def get_is_owned(self):
        return getattr(self.entity, "creator", False)

    def get_is_big(self, big_threshold: int = 1000):
        return getattr(self.entity, "participants_count", 0) > self.big_threshold

    @property
    def is_recent(self):
        return self.get_is_recent(self.recent_threshold)

    @property
    def is_owned(self):
        return self.get_is_owned()

    @property
    def is_big(self):
        return self.get_is_big(self.big_threshold)

    @property
    def name(self) -> str:
        """Generate a formatted name string for the entity.

        Format depends on entity type:
            User: "First Last @username [id]" or "First Last [id]"
            Channel: "@username Channel Title [id]" or "Channel Title [id]"
            Group: "@username Group Title [id]" or "Group Title [id]"

        Examples:
            - "@telegram Telegram News [777000]"
            - "Private Group [123456]"
            - "John Doe @johndoe [987654]"
            - "Support Group [654321]"
        """
        parts = []

        if isinstance(self.entity, User):
            # Handle users: first_name + last_name
            try:
                name_parts = []
                if hasattr(self.entity, "first_name") and self.entity.first_name:
                    name_parts.append(self.entity.first_name)
                if hasattr(self.entity, "last_name") and self.entity.last_name:
                    name_parts.append(self.entity.last_name)
                if name_parts:
                    parts.append(" ".join(name_parts))
            except AttributeError:
                pass
        else:
            # Handle channels and groups: title
            try:
                if title := self.entity.title:
                    parts.append(title)
            except AttributeError:
                pass

        # Add username if available
        try:
            if username := self.entity.username:
                parts.append(f"@{username}")
        except AttributeError:
            pass

        # Add ID in brackets (should always be available)
        try:
            parts.append(f"[{self.entity.id}]")
        except AttributeError:
            pass

        return " ".join(parts)
