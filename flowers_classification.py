from sklearn.datasets import load_iris
import math


class FlowerClassification:
    def __init__(self):
        self.iris = load_iris()

    def _distance_between_points(self, p1, p2):
        return math.sqrt(sum([ pow(p1[i] - p2[i], 2) for i, v in enumerate(p1) ]))

    def get_classification(self, data):
        distances = []

        for k, el in enumerate(self.iris.data):
            value = self._distance_between_points(el, data)
            distances.append({
                'v': value,
                'k': k
            })

        smallest = sorted(distances, key=lambda k: k['v'])[0]
        target = self.iris.target[smallest['k']]

        return self.iris['target_names'][target]
