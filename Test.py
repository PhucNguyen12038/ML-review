import numpy as np


means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]

a = 5
theta = [a]
b = np.zeros_like(theta)
c = [b]
train_ids = np.arange(1, 26)
test_ids = np.arange(26, 50)
view_ids = np.hstack((np.arange(1, 8), np.arange(14, 21)))
# print(test_ids.shape)
# print(train_ids)
# print(view_ids)
total_imgs = 4
y = np.hstack((np.zeros((total_imgs // 2,)), np.ones((total_imgs // 2,))))

y_res_id = np.array([1,2,1,3,4,1])
y_res_id = np.hstack((y_res_id, np.where(y == 1)[0]))
cls = [[0], [1]]

cls1 = [0, 1]
arr = np.zeros([2,3])
print(arr.shape)