from pynput.keyboard import Controller, Key


def parse_key(name: str) -> Key | str:
    """Zamienia nazwe z konfiguracji na klawisz pynput."""
    if name in Key.__members__:  # Key to Enum: klawisze specjalne po nazwie, np. "right"
        return Key[name]
    if len(name) == 1:           # pojedynczy znak, np. "b"
        return name
    raise ValueError(f"Nieznany klawisz w konfiguracji: {name!r}")


class KeyboardActions:
    def __init__(self, mappings: dict[str, str]):
        # Walidacja przy starcie: literowka w config.json wychodzi od razu,
        # a nie w trakcie prezentacji.
        self._keys = {gesture: parse_key(key) for gesture, key in mappings.items()}
        self._keyboard = Controller()

    def execute(self, gesture: str) -> bool:
        """Wysyla klawisz przypisany do gestu do aktywnego okna. Zwraca False, gdy brak przypisania."""
        key = self._keys.get(gesture)
        if key is None:
            return False
        self._keyboard.tap(key)
        return True