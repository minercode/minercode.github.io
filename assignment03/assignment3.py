from PIL import Image
import numpy as np
import tensorflow as tf


def init_heatconv():
    heat_window = tf.Variable([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    heat_window = tf.reshape(heat_window, shape=[3, 3, 1, 1])
    return heat_window


def expand_img(src_mat):
    img_mat = np.zeros((src_mat.shape[0] + 2, src_mat.shape[1] + 2), dtype=np.uint8)
    img_mat[1:-1, 1:-1] = src_mat
    img_mat[1:-1, 0] = src_mat[0:, 0]
    img_mat[1:-1, img_mat.shape[1] - 1] = src_mat[0:, src_mat.shape[1] - 1]
    img_mat[0, 1:-1] = src_mat[0, 0:]
    img_mat[img_mat.shape[0] - 1, 1:-1] = src_mat[src_mat.shape[0] - 1, 0:]
    return img_mat


def model(U0, ALPHA, DELTA_T):
    delta_t = DELTA_T
    alpha = ALPHA
    u_cur = expand_img(U0)

    w_constant = delta_t * alpha
    u_tensor = tf.cast(tf.Variable(u_cur), tf.float64)
    u_tensor = tf.expand_dims(u_tensor, 0)
    u_tensor = tf.expand_dims(u_tensor, 3)
    heat_window = init_heatconv()

    img_list = []
    lu_list = []
    img_list.append(u_tensor)
    list_idx = 0
    for w in w_constant:
        lu = tf.nn.conv2d(img_list[list_idx], tf.multiply(tf.cast(heat_window, tf.float64), tf.constant(w)), strides=[1, 1, 1, 1], padding='SAME')
        hl = img_list[list_idx] + lu
        img_list.append(hl)
        lu_list.append(lu)
        list_idx += 1

    return img_list


src_img = Image.open('C:/Users/CalPC_1/Pictures/lena_gray.gif').convert('L')
src_mat = np.array(src_img, 'uint8')

alpha = 0.25
delta = np.ones(128) * 0.25

img_list = model(src_mat, alpha, delta)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    res, lu = sess.run(img_list)

img_idx = [0, 1, 3, 7, 15, 31, 63, 127]

for i in img_idx:
    img_mat = res[i][0, 1:513, 1:513, 0]
    im = Image.fromarray(np.uint8(img_mat))
    im.save(str(i) + 'th iterations.jpg')





