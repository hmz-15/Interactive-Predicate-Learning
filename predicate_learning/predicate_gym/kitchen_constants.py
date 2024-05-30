from .Kitchen2D.kitchen2d.kitchen_constants import *

# countertop
TABLE_THICK = 1
SINK_POS_X = 25
SINK_W = 12
SINK_H = 10
LEFT_TABLE_WIDTH = 65
RIGHT_TABLE_WIDTH = 3
FAUCET_H = 15
FAUCET_W = 6
FAUCET_D = 0.8

# liquid
LIQUID_NAME = "water"
LIQUID_FREQUENCY = 0.2


# block-like objects
BLOCK_LIKE_OBJS = {
    "candybar": {"w": 3, "h": 3, "color": "orange", "category": "food"},
    "sausage": {"w": 3, "h": 3, "color": "red", "category": "food"},
    "bread": {"w": 3, "h": 3, "color": "yellow", "category": "food"},
    "bacon": {"w": 3, "h": 3, "color": "pink", "category": "food"},
    "fish": {"w": 3, "h": 3, "color": "purple", "category": "food"},
    "pork": {"w": 3, "h": 3, "color": "pink", "category": "food"},
    "fruitcan": {"w": 3, "h": 3, "color": "green", "category": "can"},
    "coaster": {"w": 4, "h": 2.5, "color": "brown", "category": "coaster"},
    "tomatocan": {"w": 3, "h": 3, "color": "red", "category": "can"},
    "beancan": {"w": 3, "h": 3, "color": "maroon", "category": "can"},
    "tunacan": {"w": 3, "h": 3, "color": "blue", "category": "can"},
    "shelf": {"w": 8, "h": 4, "color": "gray", "category": "shelf"},
}

# container objects
CONTAINER_OBJS = {
    "tallcup": {"w": 4, "h": 5, "d": 0.5, "shifth": 0.0, "color": "blue", "category": "cup"},
    "cup": {"w": 4, "h": 4, "d": 0.5, "shifth": 0.0, "color": "blue", "category": "cup"},
    "pot": {"w": 15, "h": 9, "d": 2.0, "shifth": 0.0, "color": "green", "category": "pot"},
    "basket": {"w": 12, "h": 6, "d": 1.0, "shifth": 0.0, "color": "purple", "category": "basket"},
}

# static polygon objects
STATIC_POLYGON_OBJS = {
    "placemat": {
        "polygon_shape": [(-3.2, 0), (3.2, 0), (3, 0.4), (-3, 0.4)],
        "w": 6.4,
        "h": 0.4,
        "color": "red",
        "category": "placemat",
    }
}

# dynamic polygon objects
DYNAMIC_POLYGON_OBJS = {
    "plate": {
        "polygon_shape": [(-2.8, 0), (2.8, 0), (3, 1.5), (-3, 1.5)],
        "w": 6,
        "h": 1.5,
        "usr_w": 6,
        "usr_h": 1.5,
        "color": "cyan",
        "category": "plate",
    },
    "baskettop": {
        "polygon_shape": [(-6, 0), (6, 0), (6, 2), (1, 2), (1, 4), (-1, 4), (-1, 2), (-6, 2)],
        "w": 12,
        "h": 4,
        "usr_w": 2,
        "usr_h": 2,
        "color": "maroon",
        "category": "potcover",
    },
}
