from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel


class Scripted(FakeMessagesListChatModel):
    """A fake chat model that replays scripted messages."""

    def bind_tools(self, tools, **kwargs):
        return self
