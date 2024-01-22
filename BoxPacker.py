"""Pack articles into boxes as evenly as possible.

Contains classes for articles, boxes, and a box packer.

Classes:
    Article
    Box
    BoxPacker

Typical usage example:
    packedBoxes = BoxPacker.pack([Article(400), Article(500), Article(600), Article(700), Article(800)], 2)
"""
from typing import *
from collections import namedtuple
import heapq

__author__ = "Robert Rosborg"
__version__ = "1.2"


class Article:
    """An article with a weight measured in grams.

    Attributes:
        __weight_in_grams: A positive integer weight of the article measured in grams.

    Notes:
        Ideally the interval of __weight_in_grams should be 100 <= __weight_in_grams <= 1000, but it's not enforced.
    """
    def __init__(self, weight_in_grams: int):
        """Inits Article with weight_in_grams.

        Args:
            weight_in_grams: The weight of the article measured in grams.

        Raises:
            ValueError: If weight_in_grams is not positive.
        """
        if weight_in_grams > 0:
            self.__weight_in_grams: int = weight_in_grams
        else:
            raise ValueError(f'Weight of article must be at least 1, not {weight_in_grams}.')

    def __str__(self) -> str:
        """String representation of Article."""
        return str(self.__weight_in_grams)

    def __repr__(self) -> str:
        """Object representation of Article."""
        return str(self.__weight_in_grams)

    def get_weight_in_grams(self) -> int:
        """Getter for __weight_in_grams."""
        return self.__weight_in_grams


class Box:
    """A box that can be packed with articles.

    Attributes:
        __box_items: A list of articles packed in the box.
        __total_weight_in_grams: An integer total weight of the content of the box.
    """
    def __init__(self, box_items: List[Article]):
        """Inits Box with box_items and the sum of all articles in box_items.

        Sets __box_items to box_items,
        and __total_weight_in_grams to the sum of the weight of all articles in __box_items.

        Args:
            box_items: A list of articles to be packed in the box.
        """
        self.__box_items: List[Article] = box_items
        self.__total_weight_in_grams: int = sum([item.get_weight_in_grams() for item in self.__box_items])

    def __str__(self) -> str:
        """String representation of Box."""
        return ', '.join(str(item) for item in self.__box_items)

    def __repr__(self) -> str:
        """Object representation of Box."""
        return '(' + str(self.__total_weight_in_grams) + ', ' + ' '.join(str(item) for item in self.__box_items) + ')'

    def add_article(self, article: Article) -> None:
        """Add article to __box_items and updates the total weight.

        Args:
            article: An articles to be packed in the box.
        """
        self.__box_items.append(article)
        self.__total_weight_in_grams += article.get_weight_in_grams()

    def get_box_items(self) -> List[Article]:
        """Getter for __box_items."""
        return self.__box_items

    def get_total_weight_in_grams(self) -> int:
        """Getter for __total_weight_in_grams."""
        return self.__total_weight_in_grams

    def combine_with_other_box(self, box2: 'Box') -> 'Box':
        """Return a new box with the combined content of this box and the content of another box.

        Args:
            box2: The other box that content will be combined from.

        Returns:
            A new box with the combined content of this box and box2.
        """
        return Box(self.__box_items + box2.get_box_items())

    @staticmethod
    def combine(box1: 'Box', box2: 'Box') -> 'Box':
        """Return a new box with the combined content of two boxes.

        Args:
            box1: The box first box that content will be combined from.
            box2: The box first box that content will be combined from.

        Returns:
            A new box with the combined content of box1 and box2.
        """
        return Box(box1.get_box_items() + box2.get_box_items())

    def __lt__(self, other: 'Box'):
        """A box is less than another box if it's content weighs less than the content of the other box
        Useful for breaking priority ties when storing boxes in a priority queue.

        Args:
            other: The box to compare this box to.

        Returns:
            True if this box weighs less than other, else False.
        """
        return self.__total_weight_in_grams < other.__total_weight_in_grams


class BoxPacker:
    """A packer for packing articles in boxes.

        Implements largest differencing method to approximate a solution to the multiway number partitioning problem.
        """
    @staticmethod
    def pack(articles: List[Article], number_of_boxes: int) -> List[Box]:
        """Pack articles in boxes as evenly as possible.

        Packs all articles in articles into number_of_boxes boxes, distributing the weight as evenly as possible.

        Args:
            articles: The articles to pack into boxes.
            number_of_boxes: The number of boxes to pack articles into.

        Returns:
            A list of boxes containing the articles in articles, with weight distributed as evenly as possible.

        Notes:
            Only calls __ldm if number_of_boxes > 1 and articles is nonempty,
            otherwise calls the Box constructor directly.

        Raises:
            ValueError: If number_of_boxes is not positive.
        """
        if not articles:
            return [Box([]) for _ in range(number_of_boxes)]

        if number_of_boxes < 1:
            raise ValueError(f'Number of boxes must be at least 1, not {number_of_boxes}.')

        if number_of_boxes == 1:
            return [Box(articles)]
        else:
            return BoxPacker.__ldm(articles, number_of_boxes)

    @staticmethod
    def __ldm(articles: List[Article], number_of_boxes: int) -> List[Box]:
        """Approximates a solution to the multiway number partitioning problem.

        Solves the problem of packing all articles in articles into number_of_boxes boxes, distributing the weight
        as evenly as possible, by approximating a solution to the multiway number partitioning problem using
        largest differencing method.

        Args:
            articles: The articles to pack into boxes.
            number_of_boxes: The number of boxes to pack articles into.

        Returns:
            A list of boxes containing the articles in articles, with weight distributed as evenly as possible.

        Notes:
            Uses a priority queue (heapq) to store the different lists of boxes, using the weight difference of
            the heaviest and the lightest box as priority. Because the priority queue in Python3 is implemented as a
            min heap and the algorithm calls for a max heap, the difference is inverted to achieve the correct behavior.

            The priority queue has the following invariant: the lists of boxes in the priority queue must be sorted
            by weight in descending order.
        """
        WeighedBoxes = namedtuple('WeighedBoxes', 'weight_difference boxes')

        def weigh_boxes(boxes: List[Box]) -> WeighedBoxes:
            """Local function for calculating the weight difference of the boxes and putting it and the boxes
            in a WeightedBoxes named tuple, so they can be inserted in the priority queue.
            """
            weights: List[int] = [box.get_total_weight_in_grams() for box in boxes]
            # min - max to get negative differance to turn min heap to max heap
            weight_difference: int = min(weights) - max(weights)
            return WeighedBoxes(weight_difference, boxes)

        def combine_boxes(boxes1: List[Box], boxes2: List[Box]) -> List[Box]:
            """Local function for combining two lists of boxes.

            Combines two lists of boxes by adding the articles of the heaviest box in the first list to the
            lightest box in the second list, then the next heaviest box in the first list to the next lightest box
            in the second list, and so on. boxes1 and boxes2 must be sorted by weight in descending order for
            this to work. It sorts the result in this way to maintain this invariant.
            """
            boxes2.reverse()
            combined_boxes: List[Box] = list(map(Box.combine, boxes1, boxes2))
            # Sort result to maintain invariant
            return sorted(combined_boxes, key=Box.get_total_weight_in_grams, reverse=True)

        # Invariant: WeightedBoxes in list must be sorted by weight in descending order
        partially_packed_boxes: List[WeighedBoxes] = []

        sorted_articles: List[Article] = sorted(articles, key=Article.get_weight_in_grams, reverse=True)
        for article in sorted_articles:
            boxes: List[Box] = [Box([]) for _ in range(number_of_boxes)]
            boxes[0].add_article(article)
            weighed_boxes: WeighedBoxes = weigh_boxes(boxes)
            partially_packed_boxes.append(weighed_boxes)

        heapq.heapify(partially_packed_boxes)

        while len(partially_packed_boxes) > 1:
            boxes_with_largest_weight_difference: WeighedBoxes = heapq.heappop(partially_packed_boxes)
            boxes_with_next_largest_weight_difference: WeighedBoxes = heapq.heappop(partially_packed_boxes)
            combined_boxes: List[Box] = combine_boxes(boxes_with_largest_weight_difference.boxes,
                                                      boxes_with_next_largest_weight_difference.boxes)
            weighed_combined_boxes: WeighedBoxes = weigh_boxes(combined_boxes)
            heapq.heappush(partially_packed_boxes, weighed_combined_boxes)

        return heapq.heappop(partially_packed_boxes).boxes
