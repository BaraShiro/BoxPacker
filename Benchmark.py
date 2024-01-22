import random
from BoxPacker import *


def greedy(articles: List[Article], number_of_boxes: int) -> List[Box]:
    """A greedy approximate solution to the multiway number partitioning problem.

        Tries to distribute the weight as evenly as possible by repeatedly putting the heaviest object in the
        lightest box.

        Args:
            articles: The articles to pack into boxes.
            number_of_boxes: The number of boxes to pack articles into.

        Returns:
            A list of boxes containing the articles in articles, with weight distributed as evenly as possible.
    """
    boxes: List[Box] = []

    for i in range(number_of_boxes):
        boxes.append(Box([]))

    for article in sorted(articles, key=Article.get_weight_in_grams, reverse=True):
        least_heavy_box = min(boxes, key=Box.get_total_weight_in_grams)
        least_heavy_box.add_article(article)

    return boxes


def run_benchmark(number_of_articles: int = 35,
                  number_of_boxes: int = 3,
                  smallest_weight: int = 100,
                  largest_weight: int = 1000) -> None:
    """Run a benchmark of two approximate solutions to the multiway number partitioning problem.

        Runs a greedy algorithm and the largest differencing method on the same data, and prints the results.

        Args:
            number_of_articles: The number of articles to be packed.
            number_of_boxes: The number of boxes to pack the articles into.
            smallest_weight: The smallest weight an article can have.
            largest_weight: The largest weight an article can have.

        Returns:
            Prints the results of benchmark.
    """
    random.seed(42)
    articles = [Article(random.randint(smallest_weight, largest_weight)) for _ in range(number_of_articles)]
    number_of_boxes = number_of_boxes

    greedy_packed_boxes = greedy(articles, number_of_boxes)
    ldm_packed_boxes = BoxPacker.pack(articles, number_of_boxes)
    greedy_weights = [box.get_total_weight_in_grams() for box in greedy_packed_boxes]
    ldm_weights = [box.get_total_weight_in_grams() for box in ldm_packed_boxes]
    print("--- Greedy algorithm ---")
    print("Article distribution:")
    print(*greedy_packed_boxes, sep="\n")
    print("Total weight:")
    print(greedy_weights)
    print("Difference between heaviest box and lightest box:")
    print(max(greedy_weights) - min(greedy_weights))

    print()

    print("--- Largest differencing method ---")
    print("Article distribution:")
    print(*ldm_packed_boxes, sep="\n")
    print("Total weight:")
    print(ldm_weights)
    print("Difference between heaviest box and lightest box:")
    print(max(ldm_weights) - min(ldm_weights))


run_benchmark()
