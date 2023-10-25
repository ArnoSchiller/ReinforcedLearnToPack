import copy
from typing import List, Tuple
import numpy as np

from gym_packing.data.item import Item
from gym_packing.data.position import Position
from gym_packing.data.snappoint import Snappoint, SnappointDirection

# describes the percentage of the bottom area required to lay on top of other item
DEFAULT_STABILITY_FACTOR = 0.75


class Bin:
    """
    Represents a bin for packing items.
    """

    def __init__(self, width: int, length: int, height: int, max_weight: 'float | None' = None,
                 stability_factor: 'float | None' = None):
        """
        Initializes a Bin object with specified dimensions and optional maximum weight.

        Args:
            width (int): The width of the bin.
            length (int): The length of the bin.
            height (int): The height of the bin.
            max_weight (float, optional): The maximum weight limit of the bin.

        """
        self.width = width
        self.length = length
        self.height = height
        self.max_weight = max_weight

        self.stability_factor = stability_factor if stability_factor is not None else DEFAULT_STABILITY_FACTOR

        self.matrix = np.zeros((height, length, width), dtype=int)
        self.packed_items: List[Item] = []

    def can_item_be_packed(self, item: Item) -> Tuple[bool, 'str | None']:
        if not item.is_packed():
            return False, f"{item.id}: Position is None."
        x, y, z = item.position.x, item.position.y, item.position.z
        if (
            x < 0
            or y < 0
            or z < 0
            or x + item.width > self.width
            or y + item.length > self.length
            or z + item.height > self.height
        ):
            return False, f"{item.id}: Item is out of bounds of the bin (containment condition)."

        if np.any(self.matrix[z: z + item.height, y: y + item.length, x: x + item.width]):
            return False, f"{item.id}: Position is already occupied (non-overlapping condition)."

        if not self._is_item_position_stable(item):
            return False, f"{item.id}: Position is not stable (stability condition)."

        return True, None

    def pack_item(self, item: Item) -> Tuple[bool, 'str | None']:
        """
        Packs an item into the bin at a valid position.

        Args:
            item (Item): The item to be packed.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating if the packing was successful
            and a string message explaining the result.

        """
        can_be_packed, info = self.can_item_be_packed(item)

        if can_be_packed:
            self.packed_items.append(item)
            x, y, z = item.position.x, item.position.y, item.position.z
            self.matrix[z: z + item.height,
                        y: y + item.length,
                        x: x + item.width] = len(self.packed_items)
        return can_be_packed, info

    def _is_item_position_stable(self, item: Item) -> bool:
        """
        Check if an item's position is stable based on the already packed items below it.

        Args:
            item (Item): The item to check for stability.

        Returns:
            bool: True if the item's position is stable, False otherwise.
        """

        # every position with z == 0 is stable
        if item.position.z == 0:
            return True

        if not item.is_packed():
            return False

        positions_below = self.matrix[item.position.z - 1,
                                      item.position.y: item.position.y + item.length,
                                      item.position.x: item.position.x + item.width]

        if np.count_nonzero(positions_below) < item.width * item.length * self.stability_factor:
            return False

        return True

    def is_packing_2d(self) -> Tuple[bool, List[str]]:
        """
        Checks if the bin is packed in 2D.

        Returns:
            Tuple[bool, List[str]]: A tuple containing a boolean indicating if the bin is packed in 2D and a string list of the two dimensions being packed.

        """
        if self.width <= 1:
            return True, ["length", "height"]
        elif self.length <= 1:
            return True, ["width", "height"]
        elif self.height <= 1:
            return True, ["width", "length"]
        return False, []

    def get_dimension_2d(self, dimensions: List[str]) -> List[int]:
        """
        Retrieves the dimensions of the bin for the specified 2D packing dimensions.

        Args:
            dimensions (List[str]): The two dimensions for the 2D packing. Should contain two strings out of ["width", "length", "height"].

        Returns:
            List[int]: A list containing the corresponding dimensions of the bin.

        Raises:
            ValueError: If the dimensions list does not contain two valid dimension strings.

        """
        if dimensions.count("width") + dimensions.count("length") + dimensions.count("height") != 2:
            raise ValueError(
                "Dimensions should contain two strings out of [width, length, height].")

        dim = []
        for d in dimensions:
            if d == "width":
                dim.append(self.width)
            if d == "length":
                dim.append(self.length)
            if d == "height":
                dim.append(self.height)
        return dim

    @property
    def volume(self) -> int:
        """
        Calculate the volume of the Bin.

        Returns:
        int: The volume of the Bin.
        """
        return int(self.width * self.length * self.height)

    def get_used_volume(self, use_percentage=False):
        """
        Calculate the used volume of the Bin.

        Args:
            use_percentage (bool, optional): Whether to return the used volume as a percentage of the total volume. 
                                            Defaults to False.

        Returns:
            float: The used volume of the Bin.
        """
        used_volume = sum([item.volume for item in self.packed_items])

        if use_percentage:
            return int(used_volume / self.volume * 100)
        return used_volume

    def get_height_map(self) -> np.ndarray:
        """
        Calculate the height map of the bin's packing matrix.

        Returns:
            numpy.ndarray: A 2D array representing the height map of the bin's packing matrix.
        """
        height_matrix = np.zeros(self.matrix.shape)
        for z in range(height_matrix.shape[0]):
            height_matrix[z] = z+1
        height_matrix *= self.matrix != 0

        height_map = np.max(height_matrix, axis=0)
        return height_map

    def get_snappoints(self, min_z: 'int | None' = None) -> List[Snappoint]:
        """
        Calculate and return the list of snap points in the bin.

        A snap point is a point in the bin where a new item could potentially be placed.
        This method calculates these points based on the current state of the bin and the items already packed.

        Note: This method is currently implemented only for 2D packing.

        Returns:
            List[Snappoint]: A list of Snappoint objects representing the snap points in the bin.

        Raises:
            NotImplementedError: If the method is called for a 3D packing scenario.
        """
        if not self.is_packing_2d():
            raise NotImplementedError(
                "get_snappoints not implemented for 3D case.")

        heightmap = copy.deepcopy(self.get_height_map()).flatten()
        if min_z is not None:
            heightmap -= min_z
            heightmap[heightmap < 0] = 0
        else:
            min_z = 0

        snappoints = []
        snappoints.append(Snappoint(
            x=0, y=0, z=int(heightmap[0]) + min_z,
            direction=SnappointDirection.RIGHT))

        for index in range(1, len(heightmap)):
            last_z = int(heightmap[index - 1]) + min_z
            next_z = int(heightmap[index]) + min_z
            if last_z != next_z:
                snappoints.append(Snappoint(
                    x=index, y=0, z=last_z, direction=SnappointDirection.LEFT))

                snappoints.append(Snappoint(
                    x=index, y=0, z=next_z, direction=SnappointDirection.RIGHT))

        snappoints.append(Snappoint(
            x=len(heightmap), y=0, z=int(heightmap[-1]) + min_z,
            direction=SnappointDirection.LEFT))

        return snappoints

    def get_center_of_gravity(self, use_volume=False) -> Position:
        """
        Calculate the center of gravity for the items packed in the bin.

        Args:
            use_volume (bool, optional): Whether to calculate the center of gravity based on item volume or weight.
                                         Defaults to False, which calculates based on item weight.

        Returns:
            Position: The calculated center of gravity.
        """
        m, x, y, z = [], [], [], []

        for item in self.packed_items:
            m.append(item.volume if use_volume else item.weight)
            x.append(item.centerpoint().x)
            y.append(item.centerpoint().y)
            z.append(item.centerpoint().z)

        m, x, y, z = np.array(m), np.array(x), np.array(y), np.array(z)

        if np.sum(m) == 0:
            return Position(x=0, y=0, z=0)

        cgx = np.sum(x*m)/np.sum(m)
        cgy = np.sum(y*m)/np.sum(m)
        cgz = np.sum(z*m)/np.sum(m)
        return Position(x=int(cgx), y=int(cgy), z=int(cgz))

    def __repr__(self):
        return f"Bin: {self.width} {self.length} {self.height} - Items{self.packed_items}"

    def __eq__(self, other):
        return self.width == other.width and self.length == other.length and self.height == other.height and self.packed_items == other.packed_items

    def __hash__(self):
        return hash((self.width, self.length, self.height, tuple(self.packed_items)))


if __name__ == '__main__':
    bin = Bin(2, 2, 2)
    item = Item("", 1, 1, 2, position=Position(0, 0, 0))
    bin.pack_item(item)

    bin2 = Bin(2, 2, 2)
    item = Item("", 1, 1, 2, position=Position(0, 0, 0))
    bin2.pack_item(item)

    print(hash(bin) == hash(bin2))
    print(hash(bin), hash(bin2))
