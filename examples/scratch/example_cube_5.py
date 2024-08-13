node_labels = ["0", "1", "2", "3", "4", "5", "6", "7",
               "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r"]

node_positions = {
    "0": (0, 0, 0),

    'a': (0.15, 0,   0),
    'b': (0.15, 0.1, 0),
    'c': (0.15, 0.1, 0.1),
    'd': (0.35, 0.1, 0.1),
    'e': (0.35, 0.1, -0.1),
    'f': (0.65, 0.1, -0.1),
    'g': (0.65, 0.1, 0),
    'h': (0.65, -0.1, 0),
    'i': (0.45, -0.1, 0),
    'j': (0.45, 0.2, 0),
    'k': (0.25, 0.2, 0),
    'l': (0.25, 0, 0),
    'm': (0.25, 0, 0.1),
    'n': (0.55, 0, 0.1),
    'o': (0.55, 0, -0.1),
    'p': (0.75, 0, -0.1),
    'q': (0.75, 0, 0),
    'r': (0.85, 0, 0),

    "1": (1, 0, 0),
    "2": (0, 1, 0),
    "3": (1, 1, 0),
    "4": (0, 0, 1),
    "5": (1, 0, 1),
    "6": (0, 1, 1),
    "7": (1, 1, 1),
}






edges = [

    ("0", "a"),
    ("a", "b"),
    ("b", "c"),
    ("c", "d"),
    ("d", "e"),
    ("e", "f"),
    ("f", "g"),
    ("g", "h"),
    ("h", "i"),
    ("i", "j"),
    ("j", "k"),
    ("k", "l"),
    ("l", "m"),
    ("m", "n"),
    ("n", "o"),
    ("o", "p"),
    ("p", "q"),
    ("q", "r"),
    ("r", "1"),

    ("0", "2"),
    ("0", "4"),
    ("1", "3"), ("1", "5"),
    ("2", "3"), ("2", "6"),
    ("3", "7"),
    ("4", "5"), ("4", "6"),
    ("5", "7"),
    ("6", "7")
]
