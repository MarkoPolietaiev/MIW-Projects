import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
#  %matplotlib inline


def generate_random_points(min_x, max_x, min_y, max_y, points_count):
    x = np.random.randint(min_x, max_x, points_count)
    y = np.random.randint(min_y, max_y, points_count)
    points = pd.DataFrame({
        'x': x,
        'y': y
    })
    return points


def assign_points_to_cetroids(points, centroids, colormap):
    distance_cols = []
    for centroid in centroids.keys():
        column_name = 'distance_to_{}'.format(centroid)
        distance_cols.append(column_name)
        dist = np.sqrt((points['x'] - centroids[centroid][0])**2+(points['y'] - centroids[centroid][1])**2)
        points[column_name] = dist
    points['closest'] = points.loc[:, distance_cols].idxmin(axis=1)
    points['closest'] = points['closest'].map(lambda x: int(x.lstrip('distance_to_')))
    points['color'] = points['closest'].map(lambda x: colormap[x])
    return points


def show_data(points, centroids, colormap):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes()
    plt.scatter(points['x'], points['y'], color=points['color'], alpha=0.3)
    for centroid in centroids.keys():
        plt.scatter(*centroids[centroid], color=colormap[centroid])
    plt.show()


def update_centroids(points, centroids):
    new_centroids = copy.deepcopy(centroids)
    for centroid in centroids.keys():
        new_centroids[centroid][0] = np.mean(points[points['closest'] == centroid]['x'])
        new_centroids[centroid][1] = np.mean(points[points['closest'] == centroid]['y'])
    return new_centroids


def show_centroid_update(points, centroids, new_centroids, colormap):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    plt.scatter(points['x'], points['y'], color=points['color'], alpha=0.3)
    for centroid in new_centroids.keys():
        plt.scatter(*new_centroids[centroid], color=colormap[centroid])
        x = centroids[centroid][0]
        y = centroids[centroid][1]
        dx = new_centroids[centroid][0] - centroids[centroid][0]
        dy = new_centroids[centroid][1] - centroids[centroid][1]
        ax.arrow(x, y, dx, dy)
    plt.show()


def generate_random_knn_point():
    knn_points = generate_random_points(-100, 100, -50, 50, 1)
    print("\nKNN Point:")
    print(knn_points.head())
    return knn_points


# calculate the distance between two points
def calculate_distance(first_point, second_point):
    distance = (first_point['x'] - second_point['x'])**2 + (first_point['y'] - second_point['y'])**2
    return np.sqrt(distance)


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for i, row in train.iterrows():
        dist = calculate_distance(test_row, row)
        distances.append((row, dist))
        distances.sort(key=lambda tup: tup[1][0])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append((distances[i][0], distances[i][1]))
    return neighbors


# get most common color(class) and assign it to our random point
def classify_our_point(neighbors, knn_point, points, centroids, colormap):
    output_values = [row[-2] for row in neighbors]
    neighbors_dataframe = pd.DataFrame(output_values)
    most_common_color = neighbors_dataframe.color.mode()
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()

    plt.scatter(points['x'], points['y'], color=points['color'], alpha=0.3)
    for centroid in centroids.keys():
        plt.scatter(*centroids[centroid], color=colormap[centroid])

    #  Point our random point
    plt.scatter(knn_point['x'], knn_point['y'], color=most_common_color)

    #  Point our nearest neighbors
    plt.scatter(neighbors_dataframe['x'], neighbors_dataframe['y'], color=neighbors_dataframe['color'], alpha=0.5)

    #  get largest distance for one of our neighbors for radius of the circle
    largest_distance = neighbors[len(neighbors) - 1][1].values

    #  create a circle to see where are our nearest neighbors
    circle = plt.Circle((knn_point['x'], knn_point['y']), radius=largest_distance[0], color='b', fill=False)
    ax = plt.gca()
    ax.add_patch(circle)

    plt.show()


def knn(points, centroids, colormap):
    knn_point = generate_random_knn_point()
    neighbors = get_neighbors(points, knn_point, 5)
    classify_our_point(neighbors, knn_point, points, centroids, colormap)


def main():
    points = generate_random_points(-100, 100, -50, 50, 300)
    print(points.head())
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(points['x'], points['y'])
    #  plt.show()
    colormap = {1: 'r', 2: 'g', 3: 'b'}
    centroids = {
        k: [np.random.randint(-100, 100), np.random.randint(-100, 100)]
        for k in colormap.keys()
    }
    assign_points_to_cetroids(points, centroids, colormap)
    #  show_data(points, centroids, colormap)  Removed redundant plot showings
    new_centroids = update_centroids(points, centroids)
    #  show_centroid_update(points, centroids, new_centroids, colormap) Removed redundant plot showings
    centroids = new_centroids
    assign_points_to_cetroids(points, centroids, colormap)
    #  show_data(points, centroids, colormap) Removed redundant plot showings
    new_centroids = update_centroids(points, centroids)
    #  show_centroid_update(points, centroids, new_centroids, colormap) Removed redundant plot showings
    centroids = new_centroids
    assign_points_to_cetroids(points, centroids, colormap)
    #  show_data(points, centroids, colormap) Removed redundant plot showings
    new_centroids = update_centroids(points, centroids)
    #  show_centroid_update(points, centroids, new_centroids, colormap) Removed redundant plot showings
    centroids = new_centroids
    assign_points_to_cetroids(points, centroids, colormap)
    #  show_data(points, centroids, colormap) Removed redundant plot showings

    knn(points, centroids, colormap)


if __name__ == "__main__":
    main()
