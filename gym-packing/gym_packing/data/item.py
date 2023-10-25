from ast import Tuple
from typing import List

from gym_packing.data.article import Article
from gym_packing.data.position import Position


class Item:
    """
    Represents an item to be packed in a container.

    Attributes:
        id (str): The identifier of the item.
        width (int): The width of the item.
        length (int): The length of the item.
        height (int): The height of the item.
        weight (float): The weight of the item.
        position (Position): The position of the item in the container.

    Methods:
        __init__(id: str, width: int, length: int, height: int, weight: float = 0.0, position: Position = None):
            Initializes an Item object with the specified attributes.

        pack(position: Position):
            Sets the position of the item in the container.

        is_packed() -> bool:
            Checks if the item is packed in a position.

        __str__() -> str:
            Returns a string representation of the item.

    """

    def __init__(self, id: str, width: int, length: int, height: int, weight: float = 0.0, position: Position = None):
        """
        Initializes an Item object with the specified attributes.

        Args:
            id (str): The identifier of the item.
            width (int): The width of the item.
            length (int): The length of the item.
            height (int): The height of the item.
            weight (float, optional): The weight of the item. Default is 0.0.
            position (Position, optional): The position of the item in the container. Default is None.

        """
        self.id = id
        self.width = width
        self.length = length
        self.height = height
        self.weight = weight
        self.position = position

    def centerpoint(self) -> Position:
        """
        Returns the centerpoint of the object.

        Returns:
            position (Position): The position of the centerpoint,

        """
        return Position(
            x=self.position.x + self.width / 2,
            y=self.position.y + self.length / 2,
            z=self.position.z + self.height / 2,
            rotation=self.position.rotation
        )

    def pack(self, position: Position):
        """
        Sets the position of the item in the container.

        Args:
            position (Position): The position object representing the coordinates and rotation of the item.

        """
        assert isinstance(
            position, Position), "This method requires a Position object as input."

        self.position = position

    def is_packed(self) -> bool:
        """
        Checks if the item is packed in a position.

        Returns:
            bool: True if the item is packed, False otherwise.

        """
        return self.position is not None

    def get_rotated_dimensions_3D(self):
        if not self.is_packed():
            return self.width, self.length, self.height

        rot_type = self.position.rotation
        if rot_type == 1:
            h, w, l = self.width, self.height, self.length
        elif rot_type == 2:
            l, w, h = self.width, self.height, self.length
        elif rot_type == 3:
            l, h, w = self.width, self.height, self.length
        elif rot_type == 4:
            h, l, w = self.width, self.height, self.length
        elif rot_type == 5:
            w, l, h = self.width, self.height, self.length
        else:  # rotation None or 0
            w, h, l = self.width, self.height, self.length

        return w, l, h

    def to_position_and_dimension_2d(self, dimensions: List[str]):
        """
        Returns the 2D position and dimensions of the item based on the specified dimensions.

        Args:
            dimensions (List[str]): The dimensions to extract from the item. Should contain two strings out of ["width", "length", "height"].

        Returns:
            Tuple[Tuple[int], Tuple[int]]: The 2D position and dimension of the item based on the input dimensions. Returns None, None if the item os not packed.

        Raises:
            ValueError: If the dimensions list does not contain two valid dimension strings.

        """
        if not self.is_packed():
            return None, None

        if dimensions.count("width") + dimensions.count("length") + dimensions.count("height") != 2:
            raise ValueError(
                "dimensions should contain two strings out of [width, length, height].")

        pos = []
        dim = []
        for d in dimensions:
            if d == "width":
                pos.append(self.position.x)
                dim.append(self.width)

            if d == "length":
                pos.append(self.position.y)
                dim.append(self.length)

            if d == "height":
                pos.append(self.position.z)
                dim.append(self.height)

        return tuple(pos), tuple(dim)

    @property
    def volume(self) -> int:
        """
        Calculate the volume of the Item.

        Returns:
        int: The volume of the Item.
        """
        return int(self.width * self.length * self.height)

    @property
    def surface(self) -> int:
        """
        Calculate the surface of the Item.

        Returns:
        int: The surface of the Item.
        """
        return int(self.width * self.length)

    def __repr__(self) -> str:
        """
        Returns a string representation of the item.

        Returns:
            str: The string representation of the item.

        """
        return f"{self.id}: width={self.width}, length={self.length}, height={self.height}, position={self.position}"

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash((self.width, self.length, self.height, self.weight, self.position.__hash__()))

    @classmethod
    def from_article(cls, article: Article) -> 'Item':
        """
        Create an Item object from an Article object.

        Args:
            article (Article): The Article object to create the Item from.

        Returns:
            Item: The created Item object.

        Example:
            >>> article = Article(article_id=1, width=10, length=20, height=5, weight=2)
            >>> item = Item.from_article(article)
            >>> print(item)
            Item(id=1, width=10, length=20, height=5, weight=2, position=None)

        """
        return cls(
            id=article.article_id,
            width=article.width,
            length=article.length,
            height=article.height,
            weight=article.weight,
            position=None
        )
