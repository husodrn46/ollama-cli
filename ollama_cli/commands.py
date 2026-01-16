from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

CommandHandler = Callable[[str], bool]


@dataclass(frozen=True)
class Command:
    name: str
    aliases: Tuple[str, ...]
    description: str
    usage: Optional[str]
    handler: CommandHandler


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: Dict[str, Command] = {}

    def register(self, command: Command) -> None:
        for key in (command.name, *command.aliases):
            self._commands[key] = command

    def get(self, name: str) -> Optional[Command]:
        return self._commands.get(name)

    def list_commands(self) -> List[Command]:
        unique = {cmd.name: cmd for cmd in self._commands.values()}
        return list(unique.values())

    def command_strings(self) -> List[str]:
        return sorted(self._commands.keys())
