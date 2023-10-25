
class Article:
    """
    Represents an article. This is used to convert to JSON and back.

    Attributes:
        article_id (str): The ID of the article.
        width (int): The width of the article.
        length (int): The length of the article.
        height (int): The height of the article.
        amount (int): The amount of the article.
        weight (float): The weight of the article.

    Methods:
        from_json(json_data: dict) -> Article:
            Creates an Article object from JSON data.

    """

    def __init__(self, article_id: str, width: int, length: int, height: int, amount: int, weight: float = 0.0):
        """
        Initializes an Article object with the specified attributes.

        Args:
            article_id (str): The ID of the article.
            width (int): The width of the article.
            length (int): The length of the article.
            height (int): The height of the article.
            amount (int): The amount of the article.
            weight (float, optional): The weight of the article. Default is 0.0.

        """
        self.article_id = article_id
        self.width = width
        self.length = length
        self.height = height
        self.amount = amount
        self.weight = weight

    def to_dict(self) -> dict:
        """
        Converts the Article object into a JSON-compatible dictionary.

        Returns:
            dict: The JSON-compatible dictionary representing the Article object.

        """
        json_data = {
            "id": self.article_id,
            "width": self.width,
            "length": self.length,
            "height": self.height,
            "weight": self.weight,
            "amount": self.amount,
        }
        return json_data

    def __repr__(self):
        return f"{self.amount}x {self.article_id} {self.width, self.length, self.height, self.weight}"

    def __eq__(self, other):
        return self.article_id == other.article_id and self.width == other.width and self.length == other.length and self.height == other.height and self.weight == other.weight and self.amount == other.amount

    @classmethod
    def from_json(cls, json_data: dict) -> 'Article':
        """
        Creates an Article object from JSON data.

        Args:
            json_data (dict): The JSON data representing the article.

        Returns:
            Article: The created Article object.

        """
        article_id = json_data["id"]
        width = json_data["width"]
        length = json_data["length"]
        height = json_data["height"]
        amount = json_data["amount"]
        weight = json_data.get("weight", 0.0)

        return cls(
            article_id=article_id,
            width=width,
            length=length,
            height=height,
            amount=amount,
            weight=weight
        )
