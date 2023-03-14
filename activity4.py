import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial import Delaunay

st.sidebar.title("Select Edit Option")
algorithm = st.sidebar.selectbox("Select Edit Style", ("Translate", "Rotate"))




def plt_basic_object_(points, counter):
    tri = Delaunay(points).convex_hull

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    S = ax.plot_trisurf(points[:,0], points[:,1], points[:,2],triangles=tri,shade=True, cmap=cm.seismic,lw=0.5)

    ax.set_xlim3d(-10, 10)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(-10, 10)
    if (counter == 1):
        plt.title("Octahedron")
    elif (counter == 2):
        plt.title("Pyramid")
    elif (counter == 3):
        plt.title("Hexagonal Prism")

    return fig




#option for translate
if algorithm == "Translate":
    def _octahedron_(bottom_lower=(0,0,0,), side_length=5):
        bottom_lower = np.array(bottom_lower)

        points = np.vstack([
            bottom_lower + [0, 0, side_length/2],
            bottom_lower + [0, side_length/2, 0],
            bottom_lower + [side_length/2, 0, 0],
            bottom_lower + [0, -side_length/2, 0],
            bottom_lower + [-side_length/2, 0, 0],
            bottom_lower + [0, 0, -side_length/2],
        ])
        return points

    init_octahedron = _octahedron_(bottom_lower=(0,0,0))
    points_heart = tf.constant(init_octahedron, dtype=tf.float32)
    counter = 1
    st.title("Image Translate Octahedron")
    x = st.slider("Enter for x:", -10, 10, 0, step=1,key='my_slider1')
    y = st.slider("Enter for y:", -10, 10, 0,step=1,key='my_slider2')
    z = st.slider("Enter for z:", -10, 10, 0,step=1,key='my_slider3')

    translation = tf.constant([x, y, z], dtype=tf.float32)

    translated_points = points_heart + translation

    fig1 = plt_basic_object_(translated_points.numpy(), counter)
    st.pyplot(fig1)




# for pyramid

    def _pyramid_(bottom_center=(0, 0, 0)):
        bottom_center = np.array(bottom_center) 

        points = np.vstack([
        bottom_center + [-3, -3, 0],
        bottom_center + [-3, +3, 0],
        bottom_center + [+3, -3, 0],
        bottom_center + [+3, +3, 0],
        bottom_center + [0, 0, +5]
        ])

        return points

    init_pyramid = _pyramid_(bottom_center=(0,0,0))
    points_pyramid2 = tf.constant(init_pyramid, dtype=tf.float32)
    counter = 2
    st.title("Image Translate Pyramid")
    x = st.slider("Enter for x:", -10, 10, 0, step=1,key='my_slider4')
    y = st.slider("Enter for y:", -10, 10, 0, step=1,key='my_slider5')
    z = st.slider("Enter for z:", -10, 10, 0, step=1,key='my_slider6')

    translation = tf.constant([x, y, z], dtype=tf.float32)

    translated_points = points_pyramid2 + translation

    fig2 = plt_basic_object_(translated_points.numpy(), counter)
    st.pyplot(fig2)




#fucntion for hexagonal_prism


    def _hexagonal_prism_(bottom_lower=(0, 0, 0), side_length=5, height=5):             
        bottom_lower = np.array(bottom_lower)

        a = side_length/2
        b = np.sqrt(3)*side_length/2
        h = height
        points = np.vstack([
            bottom_lower + [a, 0, 0],
            bottom_lower + [a/2, b/2, 0],
            bottom_lower + [-a/2, b/2, 0],
            bottom_lower + [-a, 0, 0],
            bottom_lower + [-a/2, -b/2, 0],
            bottom_lower + [a/2, -b/2, 0],
            bottom_lower + [a, 0, h],
            bottom_lower + [a/2, b/2, h],
            bottom_lower + [-a/2, b/2, h],
            bottom_lower + [-a, 0, h],
            bottom_lower + [-a/2, -b/2, h],
            bottom_lower + [a/2, -b/2, h]
        ])
        return points

    init_hexagonal_prism = _hexagonal_prism_(bottom_lower=(0,0,0))
    points_hexagonal_prism = tf.constant(init_hexagonal_prism, dtype=tf.float32)
    counter = 3
    st.title("Image Translate Hexagonal Prism")
    x = st.slider("Enter for x:", -10, 10, 0, step=1,key='my_slider7')
    y = st.slider("Enter for y:", -10, 10, 0, step=1,key='my_slider8')
    z = st.slider("Enter for z:", -10, 10, 0, step=1,key='my_slider9')

    translation = tf.constant([x, y, z], dtype=tf.float32)

    translated_points = points_hexagonal_prism + translation

    fig3 = plt_basic_object_(translated_points.numpy(), counter)
    st.pyplot(fig3)








# Option for rotation
elif algorithm == "Rotate":
    def _octahedron_(bottom_lower=(0,0,0,), side_length=5):
        bottom_lower = np.array(bottom_lower)

        points = np.vstack([
            bottom_lower + [0, 0, side_length/2],
            bottom_lower + [0, side_length/2, 0],
            bottom_lower + [side_length/2, 0, 0],
            bottom_lower + [0, -side_length/2, 0],
            bottom_lower + [-side_length/2, 0, 0],
            bottom_lower + [0, 0, -side_length/2],
        ])
        return points

    init_octahedron = _octahedron_(bottom_lower=(0,0,0))
    points_heart = tf.constant(init_octahedron, dtype=tf.float32)
    counter = 1
    st.title("Image Rotate Octahedron")
    x = st.slider("Rotate around X axis:", -180, 180, 0, step=1,key='my_slider10')
    y = st.slider("Rotate around Y axis:", -180, 180, 0, step=1,key='my_slider11')
    z = st.slider("Rotate around Z axis:", -180, 180, 0, step=1,key='my_slider12')

    rotation_x = tf.constant([[1, 0, 0],
                            [0, np.cos(np.deg2rad(x)), -np.sin(np.deg2rad(x))],
                            [0, np.sin(np.deg2rad(x)), np.cos(np.deg2rad(x))]], dtype=tf.float32)

    rotation_y = tf.constant([[np.cos(np.deg2rad(y)), 0, np.sin(np.deg2rad(y))],
                            [0, 1, 0],
                            [-np.sin(np.deg2rad(y)), 0, np.cos(np.deg2rad(y))]], dtype=tf.float32)

    rotation_z = tf.constant([[np.cos(np.deg2rad(z)), -np.sin(np.deg2rad(z)), 0],
                            [np.sin(np.deg2rad(z)), np.cos(np.deg2rad(z)), 0],
                            [0, 0, 1]], dtype=tf.float32)

    rotated_points = tf.matmul(points_heart, rotation_x)
    rotated_points = tf.matmul(rotated_points, rotation_y)
    rotated_points = tf.matmul(rotated_points, rotation_z)

    fig1 = plt_basic_object_(rotated_points, counter)
    st.pyplot(fig1)






#for pyramid rotate
    def _pyramid_(bottom_lower=(0, 0, 0)):
        bottom_lower = np.array(bottom_lower) 

        points = np.vstack([
        bottom_lower + [-3, -3, 0],
        bottom_lower + [-3, +3, 0],
        bottom_lower + [+3, -3, 0],
        bottom_lower + [+3, +3, 0],
        bottom_lower + [0, 0, +5]
        ])

        return points

    def rotate_obj(points, x_angle, y_angle, z_angle):
        x_angle = float(x_angle)
        y_angle = float(y_angle)
        z_angle = float(z_angle)

        rotation_matrix = tf.stack([
            [tf.cos(x_angle) * tf.cos(y_angle), 
            tf.cos(x_angle) * tf.sin(y_angle) * tf.sin(z_angle) - tf.sin(x_angle) * tf.cos(z_angle),
            tf.cos(x_angle) * tf.sin(y_angle) * tf.cos(z_angle) + tf.sin(x_angle) * tf.sin(z_angle)],

            [tf.sin(x_angle) * tf.cos(y_angle),
            tf.sin(x_angle) * tf.sin(y_angle) * tf.sin(z_angle) + tf.cos(x_angle) * tf.cos(z_angle),
            tf.sin(x_angle) * tf.sin(y_angle) * tf.cos(z_angle) - tf.cos(x_angle) * tf.sin(z_angle)],

            [-tf.sin(y_angle),
            tf.cos(y_angle) * tf.sin(z_angle),
            tf.cos(y_angle) * tf.cos(z_angle)]
        ])

        rotated_points = tf.matmul(tf.cast(points, tf.float32), tf.cast(rotation_matrix, tf.float32))

        return rotated_points

    init_pyramid = _pyramid_(bottom_lower=(0,0,0))
    points_pyramid = tf.constant(init_pyramid, dtype=tf.float32)
    counter = 3
    st.title("Image Rotate Pyramid")
    x = st.slider("Enter for x:", -180, 180, 0, step=1,key='my_slider13')
    y = st.slider("Enter for y:", -180, 180, 0, step=1,key='my_slider14')
    z = st.slider("Enter for z:", -180, 180, 0, step=1,key='my_slider15')

    with tf.compat.v1.Session() as session:
        points_pyramid2 = tf.constant(_pyramid_(bottom_lower=(0, 0, 0)), dtype=tf.float32)
        rotated_pyramid = session.run(rotate_obj(points_pyramid2, x/180*np.pi, y/180*np.pi, z/180*np.pi))

    fig2 = plt_basic_object_(rotated_pyramid, counter)
    st.pyplot(fig2)








    # for hexagonal prism rotate

    def _hexagonal_prism_(bottom_lower=(0, 0, 0), side_length=5, height=5):             
            bottom_lower = np.array(bottom_lower)

            a = side_length/2
            b = np.sqrt(3)*side_length/2
            h = height
            points = np.vstack([
                bottom_lower + [a, 0, 0],
                bottom_lower + [a/2, b/2, 0],
                bottom_lower + [-a/2, b/2, 0],
                bottom_lower + [-a, 0, 0],
                bottom_lower + [-a/2, -b/2, 0],
                bottom_lower + [a/2, -b/2, 0],
                bottom_lower + [a, 0, h],
                bottom_lower + [a/2, b/2, h],
                bottom_lower + [-a/2, b/2, h],
                bottom_lower + [-a, 0, h],
                bottom_lower + [-a/2, -b/2, h],
                bottom_lower + [a/2, -b/2, h]
            ])
            return points

    def rotate_obj(points, angle):
        angle_x, angle_y, angle_z = angle
        rotation_matrix_x = tf.stack([[1, 0, 0], [0, tf.cos(angle_x), tf.sin(angle_x)], [0, -tf.sin(angle_x), tf.cos(angle_x)]])
        rotation_matrix_y = tf.stack([[tf.cos(angle_y), 0, -tf.sin(angle_y)], [0, 1, 0], [tf.sin(angle_y), 0, tf.cos(angle_y)]])
        rotation_matrix_z = tf.stack([[tf.cos(angle_z), tf.sin(angle_z), 0], [-tf.sin(angle_z), tf.cos(angle_z), 0], [0, 0, 1]])
        
        rotation_matrix = tf.matmul(tf.matmul(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)

        rotate_prism = tf.matmul(tf.cast(points, tf.float32), tf.cast(rotation_matrix, tf.float32))

        return rotate_prism

    init_hexagonal_prism = _hexagonal_prism_(bottom_lower=(0,0,0))
    points_hexagonal_prism = tf.constant(init_hexagonal_prism, dtype=tf.float32)
    counter = 3
    st.title("Image Rotate Hexagonal Prism")
    x = st.slider("Enter for x:", -180, 180, 0, step=1,key='my_slider16')
    y = st.slider("Enter for y:", -180, 180, 0, step=1,key='my_slider17')
    z = st.slider("Enter for z:", -180, 180, 0, step=1,key='my_slider18')

    rotated_points = rotate_obj(points_hexagonal_prism, [x/180*np.pi, y/180*np.pi, z/180*np.pi])

    fig3 = plt_basic_object_(rotated_points, counter)
    st.pyplot(fig3)
