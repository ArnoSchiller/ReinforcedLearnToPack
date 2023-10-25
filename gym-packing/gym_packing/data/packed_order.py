import json
from typing import Dict, List

from gym_packing.data.article import Article
from gym_packing.data.bin import Bin
from gym_packing.data.item import Item
from gym_packing.data.packing_variant import PackingVariant
from gym_packing.data.position import Position


class PackedOrder:
    """
    Represents a packed order containing multiple packing variants.

    Attributes:
        order_id (str): The ID of the order.
        packing_variants (List[PackingVariant]): A list of packing variants associated with the order.
    """

    def __init__(self, order_id: str):
        """
        Initializes a new instance of the PackedOrder class.

        Args:
            order_id (str): The ID of the order.
        """
        if not isinstance(order_id, str):
            raise TypeError("order_id should be of type string.")

        self.order_id = order_id
        self.packing_variants: List[PackingVariant] = []

    def add_packing_variant(self, packing_variant: PackingVariant):
        """
        Adds a packing variant to the packed order.

        Args:
            packing_variant (PackingVariant): The packing variant to be added.
        """
        if not isinstance(packing_variant, PackingVariant):
            raise TypeError(
                "Packing variant should be of class PackingVariant.")

        self.packing_variants.append(packing_variant)

    def get_articles_list(self) -> List[Article]:
        """
        Get a list of articles based on the packed items in the packing variants and unpacked items.

        Returns:
            List[Article]: A list of articles with aggregated information.

        Example:
            Suppose the packing variants contain the following items:
            - item1 (width=10, length=10, height=10, weight=1.0)
            - item2 (width=20, length=20, height=20, weight=2.0)
            - item1 (width=10, length=10, height=10, weight=1.0)

            The function will return the following list of articles:
            [
                Article(article_id="item1", width=10, length=10, height=10, amount=2, weight=2.0),
                Article(article_id="item2", width=20, length=20, height=20, amount=1, weight=2.0)
            ]
        """
        items = []
        for variant in self.packing_variants:
            for bin in variant.bins:
                for item in bin.packed_items:
                    items.append(item)
            for item in variant.unpacked_items:
                items.append(item)

        articles = []
        for item in items:
            filtered = list(
                filter(lambda x: x.article_id == item.id, articles))
            if len(filtered) > 0:
                index = articles.index(filtered[0])
                articles[index].amount += 1
            else:
                articles.append(
                    Article(
                        article_id=item.id,
                        width=item.width,
                        length=item.length,
                        height=item.height,
                        amount=1,
                        weight=item.weight,
                    )
                )
        return articles

    def __repr__(self):
        return f"PackedOrder({self.order_id}, {self.packing_variants})"

    def __eq__(self, other: object) -> bool:
        return self.order_id == other.order_id and self.packing_variants == other.packing_variants

    def to_dict(self, as_string=True) -> str:
        """
        Convert the PackedOrder object to JSON format.

        Returns:
            str: JSON representation of the PackedOrder.
        """
        data = {
            "order_id": self.order_id,
            "packing_variants": [],
            "articles": [article.to_dict() for article in self.get_articles_list()]
        }

        for variant in self.packing_variants:
            variant_data = []

            for idx, bin in enumerate(variant.bins):
                positions = []
                for item in bin.packed_items:
                    positions.append({
                        "article_id": item.id,
                        "x": item.position.x,
                        "y": item.position.y,
                        "z": item.position.z,
                        "rotation": item.position.rotation,
                        "centerpoint_x": item.centerpoint().x,
                        "centerpoint_y": item.centerpoint().y,
                        "centerpoint_z": item.centerpoint().z
                    })

                variant_data.append({
                    "colli": idx + 1,
                    "colli_total": len(variant.bins),
                    "colli_dimension": {
                        "width": bin.width,
                        "length": bin.length,
                        "height": bin.height
                    },
                    "positions": positions
                })

            data["packing_variants"].append(variant_data)

        if as_string:
            return json.dumps(data, indent=4)
        return data

    def write_to_file(self, file_path: str):
        """
        Write the PackedOrder object to a JSON file.

        Args:
            file_path (str): The path of the file to write the JSON data to.
        """
        json_data = self.to_dict(as_string=True)
        with open(file_path, "w") as file:
            file.write(json_data)

    @classmethod
    def load_from_json(cls, json_str: str) -> 'PackedOrder':
        """
        Load a PackedOrder object from a JSON string.

        Args:
            json_str (str): The JSON string representing the PackedOrder object.

        Returns:
            PackedOrder: The loaded PackedOrder object.
        """
        data = json.loads(json_str)
        return PackedOrder.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict) -> 'PackedOrder':
        """
        Load a PackedOrder object from a dict.

        Args:
            data (dict): The dict representing the PackedOrder object.

        Returns:
            PackedOrder: The loaded PackedOrder object.
        """
        order_id = data.get("order_id", None)
        if order_id is None:
            raise ValueError("Invalid JSON format. Missing 'order_id' field.")

        packed_order = cls(order_id)

        article_dict = {}
        for a in data.get("articles", []):
            article = Article.from_json(a)
            article_dict[article.article_id] = article

        variants_data = data.get("packing_variants", [])
        for variant_data in variants_data:
            variant = PackingVariant()
            for bin_data in variant_data:

                colli_dim = bin_data.get("colli_dimension")
                bin = Bin(
                    colli_dim["width"], colli_dim["length"], colli_dim["height"])

                for pos_data in bin_data.get("positions", []):
                    article = article_dict[pos_data["article_id"]]
                    pos = Position(
                        x=int(pos_data["x"]),
                        y=int(pos_data["y"]),
                        z=int(pos_data["z"]),
                        rotation=pos_data["rotation"]
                    )
                    item = Item(
                        id=article.article_id,
                        width=article.width,
                        length=article.length,
                        height=article.height,
                        weight=article.weight,
                        position=pos
                    )
                    is_packed, _ = bin.pack_item(item)

                    if not is_packed:
                        variant.unpacked_items.append(item)
                variant.add_bin(bin)

            packed_order.add_packing_variant(variant)

        return packed_order

    @classmethod
    def load_from_file(cls, file_path: str) -> 'PackedOrder':
        """
        Load a PackedOrder object from a JSON file.

        Args:
            file_path (str): The path of the JSON file.

        Returns:
            PackedOrder: The loaded PackedOrder object.
        """
        with open(file_path, "r") as file:
            json_data = file.read()
        return cls.load_from_json(json_data)
