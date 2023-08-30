import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
Points = {'Blue': [[3,2], [2,4], [2,3], [1,2], [2,1]],
          'Red': [[6,6], [5,4], [3,6], [5,6], [5,3]]}

NewPoint = [4,2]

class NN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, Points):
        self.Points = Points

    def Distance(self, p, q):
        return np.sqrt(np.sum(np.array(p) - np.array(q)) ** 2)

    def Prediction(self, NewPoint):
        Distances = []

        for category in self.Points:
            for Point in self.Points[category]:
                Distance = self.Distance(Point, NewPoint)
                Distances.append([Distance, category])

        categories = [category[1] for category in sorted(Distances)[:self.k]]
        Value = Counter(categories).most_common(1)[0][0]
        return Value


clf = NN(k=3)
clf.fit(Points)
print(clf.Prediction(NewPoint))


ax = plt.subplot()
ax.grid(True, color='#323232')

ax.set_facecolor('black')
ax.figure.set_facecolor('#121212')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

[ax.scatter(Point[0], Point[1], color='#104DCA', s=60) for Point in Points['Blue']]
[ax.scatter(Point[0], Point[1], color='#990000', s=60) for Point in Points['Red']]

new_class = clf.Prediction(NewPoint)
color = '#990000' if new_class == 'Red' else '#104DCA'
ax.scatter(NewPoint[0], NewPoint[1], color=color, marker='*', s=200, zorder=100)

[ax.plot([NewPoint[0], Point[0]], [NewPoint[1], Point[1]], color='#104DCA', linestyle='--', linewidth=1) for Point in Points['Blue']]
[ax.plot([NewPoint[0], Point[0]], [NewPoint[1], Point[1]], color='#990000', linestyle='--', linewidth=1) for Point in Points['Red']]

plt.show()