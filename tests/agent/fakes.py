from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel


class Scripted(FakeMessagesListChatModel):
    """A fake chat model that replays scripted messages."""

    def bind_tools(self, tools, **kwargs):
        return self


class Recording(Scripted):
    """A scripted model that keeps the rules it was given."""

    rules: list = []

    def _generate(self, messages, *args, **kwargs):
        self.rules.append(messages[0].text)
        return super()._generate(messages, *args, **kwargs)


class Broken(Scripted):
    """A model whose provider refuses every call, like an account out of credits."""

    def _generate(self, *args, **kwargs):
        raise RuntimeError("402: add usage credits")
