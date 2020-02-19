import numpy as np
import matplotlib.pyplot as plt


# def test_reference():
#     x = np.linspace(0, 22.375, 100)
#     y = -2.42636 * x**2+ 0.11043 * x**3
#     print(x, y)
#     plt.plot(x, y)
#     ax = plt.gca()
#     ax.set_aspect(1)
#     plt.show()
#
# def test_plot():
#     plt.gca().add_patch(plt.Rectangle((-18, -18), 36, 36, facecolor='none'))
#
#     plt.plot([0, 1], [0, 1], linestyle='--')
#     plt.show()

import bezier
# x=9
# nodes1 = np.asfortranarray([[0.0, 0.0, -18+x, -18.0], [0.0, x, 18, 18]])  # [[0.0, 0.0, -18.0], [0.0, 18.0, 18.0]]
# curve1 = bezier.Curve(nodes1, degree=3)
# s_vals = np.linspace(0.0, 1.0, 30)
# data1=curve1.evaluate_multi(s_vals)
# x33=data1[0]
# y33=data1[1]
plt.axis('equal')
#
#
nodes2 = np.asfortranarray([[3.75/2, 3.75/2, -18+10, -18-2], [-18-10, -18+10, 3.75/2, 3.75/2]])  # [[0.0, 0.0, -18.0], [0.0, 18.0, 18.0]]
curve2 = bezier.Curve(nodes2, degree=3)
s_vals = np.linspace(0, 1.0, 30)
data2=curve2.evaluate_multi(s_vals)
x332=data2[0]
y332=data2[1]

y=9-3.75*3/4
nodes3 = np.asfortranarray([[3.75/2, 3.75/2, -18+10, -18-2], [-18-10, -18+10, 3.75*3/2, 3.75*3/2]])  # [[0.0, 0.0, -18.0], [0.0, 18.0, 18.0]]
curve3 = bezier.Curve(nodes3, degree=3)
s_vals = np.linspace(0, 1.0, 30)
data3=curve3.evaluate_multi(s_vals)
x333=data3[0]
y333=data3[1]
print(x333, y333, type(data3))
# plt.plot(x33, y33, color="#800080", linewidth=2.0, linestyle="-", label="y2")
plt.plot(x332, y332, color="#100080", linewidth=2.0, linestyle="-", label="1")
plt.plot(x333, y333, color="#200080", linewidth=2.0, linestyle="-", label="1")

plt.plot([0, 0], [-18, -30], color='black')
plt.plot([3.75, 3.75], [-18, -30], color='black')

plt.plot([-18, -30], [3.75*2, 3.75*2], color='black')
plt.plot([-18, -30], [3.75, 3.75], color='black')
plt.plot([-18, -30], [0, 0], color='black')

plt.show()

# import tensorflow as tf
# x = tf.Variable(5.0)
# z = tf.Variable(6.0)
# with tf.GradientTape() as tape:
#     arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 5]
#     t2 = tf.cast(x, dtype=tf.int32)
#     y = arr[t2]
# a = tape.gradient(y, x)
# print(a)


# x = tf.placeholder(dtype=tf.int32, name='x')
# arr = tf.constant([1,2,3,4,5,6,7,8,9,2,3,5])
# y = arr[x]
# print(y)
# op = tf.gradients(y, x)
# w = tf.Variable(2.0, name='weight')
# new_w = w.assign(op)
# sess = tf.Session()
# print(sess.run(new_w, feed_dict={x: 3}))

# def test_plot():
#     plt.plot([-13.923711809202135, -12.313886384461737], [1.450971841142493, 2.950459120801578], color='black')
#     plt.plot([-12.313886384461737, -9.042277774296462], [2.950459120801578, -0.561887260450197], color='black')
#     plt.plot([-10.65210319903686, -9.042277774296462], [-2.0613745401092816, -0.561887260450197], color='black')
#     plt.plot([-13.923711809202135, -10.65210319903686], [1.450971841142493, -2.0613745401092816], color='black')
#     plt.plot(-11.482994791749299, 0.444542290346148, marker='x', color='red')
#
#     plt.axis('equal')
#     plt.show()


#
# if __name__ == "__main__":
#     test_plot()