"""
Python attrs testing.
"""

# attrs

import inspect
import random
import string
from pprint import pprint

import attr


def generate_id() -> str:
    return "".join(random.choices(string.ascii_uppercase, k=12))


@attr.s()
class Person:
    name: str = attr.ib(validator=attr.validators.instance_of(str))
    address: str = attr.ib(default="", converter=str)
    active: bool = attr.ib(default=True)
    email_addresses: list[str] = attr.ib(factory=list)
    id: str = attr.ib(factory=generate_id, init=False)
    _search_string: str = attr.ib(init=False, repr=False)

    def __post_init__(self) -> None:
        self._search_string = f"{self.name} {self.address}"


def main() -> None:
    person = Person(name="Haseeb", address="Guro-gu")
    pprint(person)
    pprint(person.__dict__)
    pprint(inspect.getmembers(Person, inspect.isfunction))


if __name__ == "__main__":
    main()
