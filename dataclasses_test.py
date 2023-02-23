"""
Python dataclasses testing.
"""

# Dataclasses
"""
Data classes automatically generate various methods for us
like the '__repr__' method.

Also our code looks cleaner.
"""

import inspect
import random
import string
from dataclasses import dataclass, field
from pprint import pprint


def generate_id() -> str:
    return "".join(random.choices(string.ascii_uppercase, k=12))


@dataclass(frozen=True)
class Person:
    name: str
    address: str
    active: bool = True
    email_addresses: list[str] = field(default_factory=list)
    id: str = field(init=False, default_factory=generate_id)
    # _search_string: str = field(init=False, repr=False)

    # def __post_init__(self) -> None:
    #     self._search_string = f"{self.name} {self.address}"


def main() -> None:
    person = Person(name="Haseeb", address="Guro-gu")
    pprint(person)
    pprint(person.__dict__)
    pprint(inspect.getmembers(Person, inspect.isfunction))


if __name__ == "__main__":
    main()
