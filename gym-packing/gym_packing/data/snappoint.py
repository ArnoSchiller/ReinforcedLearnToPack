from enum import Enum


class SnappointDirection(Enum):
    LEFT = 1
    RIGHT = 2

    def change(self):
        if self == SnappointDirection.LEFT:
            return SnappointDirection.RIGHT
        return SnappointDirection.LEFT


class Snappoint:
    """
    Represents a position in 3D space where a item can be packed.

    Attributes:
        x (int): The X-coordinate of the position.
        y (int): The Y-coordinate of the position.
        z (int): The Z-coordinate of the position.
        direction (SnappointDirection): Which corner of the item should overlap with this position.

    """

    def __init__(self, x: int, y: int, z: int, direction: SnappointDirection):
        """
        Initializes a direction object with the specified coordinates and packing direction.

        Args:
            x (int): The X-coordinate of the position.
            y (int): The Y-coordinate of the position.
            z (int): The Z-coordinate of the position.
            direction (SnappointDirection): Which corner of the item should overlap with this position.

        """
        self.x = x
        self.y = y
        self.z = z
        self.direction = direction

    def __repr__(self):
        return f"Snappoint: {self.x, self.y, self.z, self.direction}"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z and self.direction == other.direction
