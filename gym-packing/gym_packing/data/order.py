import json
import os
from typing import List

from gym_packing.data.article import Article
from gym_packing.data.item import Item


class Supply:
    """
    Represents a supply.

    Attributes:
        supply_id (str): The ID of the supply.
        amount (int): The amount of the supply.

    Methods:
        from_json(json_data: dict) -> Supply:
            Creates a Supply object from JSON data.

    """

    def __init__(self, supply_id: str, amount: int):
        """
        Initializes a Supply object with the specified attributes.

        Args:
            supply_id (str): The ID of the supply.
            amount (int): The amount of the supply.

        """
        self.supply_id = supply_id
        self.amount = amount

    @classmethod
    def from_json(cls, json_data: dict) -> 'Supply':
        """
        Creates a Supply object from JSON data.

        Args:
            json_data (dict): The JSON data representing the supply.

        Returns:
            Supply: The created Supply object.

        """
        supply_id = json_data.get("id", None)
        amount = json_data.get("amount", None)

        return cls(supply_id, amount)


class Order:
    """
    Represents an packing order.

    Attributes:
        order_id (str): The ID of the order.
        articles (List[Article]): The list of articles in the order.
        supplies (List[Supply]): The list of supplies in the order.

    Methods:
        to_item_list() -> List[Item]:
            Converts the articles in the order to a list of Item objects.
        from_json_file(file_path: str) -> Order:
            Creates an Order object from a JSON file.
        from_json(json_data: dict) -> Order:
            Creates an Order object from JSON data.

    """

    def __init__(self, order_id: str, articles: List[Article], supplies: List[Supply] = []):
        """
        Initializes an Order object with the specified attributes.

        Args:
            order_id (str): The ID of the order.
            articles (List[Article]): The list of articles in the order.
            supplies (List[Supply], optional): The list of supplies in the order. Default is an empty list.

        """
        self.order_id = order_id
        self.articles = articles
        self.supplies = supplies

    def to_item_list(self) -> List[Item]:
        """
        Converts the articles in the order to a list of Item objects.

        Returns:
            List[Item]: The list of Item objects.

        """
        return [
            Item(
                id=article.article_id,
                width=article.width,
                length=article.length,
                height=article.height,
                weight=article.weight,
                position=None
            )
            for article in self.articles
        ]

    def to_dict(self) -> dict:
        """
        Transforms the Order object into a JSON-compatible dictionary.

        Returns:
            dict: The JSON-compatible dictionary representing the Order.

        """
        json_data = {
            "order_id": self.order_id,
            "articles": [article.to_dict() for article in self.articles],
            "supplies": [supply.to_dict() for supply in self.supplies]
        }
        return json_data

    def __repr__(self):
        return f"{self.order_id} ({self.articles})"

    def __eq__(self, other):
        return self.order_id == other.order_id and self.articles == other.articles

    @classmethod
    def from_json_file(cls, file_path: str) -> 'Order':
        """
        Creates an Order object from a JSON file.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            Order: The created Order object.

        """
        if not os.path.exists(file_path):
            return None

        with open(file_path, "r") as file:
            json_data = json.load(file)
        return cls.from_json(json_data)

    @classmethod
    def from_json(cls, json_data: dict) -> 'Order':
        """
        Creates an Order object from JSON data.

        Args:
            json_data (dict): The JSON data representing the order.

        Returns:
            Order: The created Order object.

        Raises:
            ValueError: If the order does not have at least one article.

        """
        order_id = json_data["order_id"]
        articles = [Article.from_json(article_data)
                    for article_data in json_data.get("articles", [])]
        supplies = [Supply.from_json(supply_data)
                    for supply_data in json_data.get("supplies", [])]

        if len(articles) < 1:
            raise ValueError("Order must have at least one article.")

        return cls(order_id, articles, supplies)
