import numpy as np
from cosine_analysis import average_cosdist
import scipy.spatial.distance

x1 = np.array([[1, 2, 3, 4]])
y1 = np.array([[1, 2, 3, 4]])
elm1 = np.array([scipy.spatial.distance.cosine(np.array([1]), np.array([1]))])
elm2 = np.array([scipy.spatial.distance.cosine(np.array([2]), np.array([2]))])
elm3 = np.array([scipy.spatial.distance.cosine(np.array([3]), np.array([3]))])
elm4 = np.array([scipy.spatial.distance.cosine(np.array([4]), np.array([4]))])
cos_dist1 = np.array([elm1, elm2, elm3, elm4])
assert np.array_equal(average_cosdist(x1, y1), cos_dist1)

x2 = np.array([[10]])
y2 = np.array([[-10]])
cos_dist2 = np.array([[scipy.spatial.distance.cosine(np.array([10]), np.array([-10]))]])
assert np.array_equal(average_cosdist(x2, y2), cos_dist2)

x3 = np.array([[0.033243], [0.356235], [0.000001]])
y3 = np.array([[0.42355, -0.1232], [0.34523, 0.000000421], [0.1200001, -0.51840124]])
elm1 = np.array([scipy.spatial.distance.cosine(np.array([0.033243, 0.356235, 0.000001]), np.array([0.42355, 0.34523, 0.1200001]))])
elm2 = np.array([scipy.spatial.distance.cosine(np.array([0.033243, 0.356235, 0.000001]), np.array([-0.1232, 0.000000421, -0.51840124]))])
cos_dist3 = np.array([(np.array(elm1) + np.array(elm2)) / 2])
assert np.array_equal(average_cosdist(x3, y3), cos_dist3)

x3 = np.array([[-3], [-2], [-1]])
y3 = np.array([[1], [2], [3]])
cos_dist3 = np.array([[scipy.spatial.distance.cosine(np.array([-3, -2, -1]), np.array([1, 2, 3]))]])
assert np.array_equal(average_cosdist(x3, y3), cos_dist3)

x4 = np.array([[78.220], [-571.72], [90086.5], [922.2256]])
y4 = np.array([[11235], [33.8317], [3.00000], [32821]])
cos_dist4 = np.array([[scipy.spatial.distance.cosine(np.array([78.220, -571.72, 90086.5, 922.2256]), np.array([11235, 33.8317, 3.00000, 32821]))]])
assert np.array_equal(average_cosdist(x4, y4), cos_dist4)
