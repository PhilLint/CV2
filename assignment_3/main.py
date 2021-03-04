"""
Main Assignment 3 file
"""
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

p = Path(__file__).resolve().parents[1]
sys.path.append(str(p))

from assignment_3.utils.mesh_to_png import mesh_to_png, file_to_mesh
from assignment_3.morphable_model import morphable_model
from assignment_3.pinhole_camera_model import *

FILE_MODEL = p / 'assignment_3/data/model2017-1_face12_nomouth.h5'
FILE_LANDMARKS = p / 'assignment_3/data/Landmarks68_model2017-1_face12_nomouth.anl'
OUT = p / 'assignment_3/out/'
WIDTH = 12
HEIGHT = 8


def plot_base():
    """
    Plots the base model (identity and expression)
    """
    mesh = file_to_mesh(FILE_MODEL)
    mesh_to_png(OUT / 'base.png', mesh)


def section_2_figures(no_figures):
    """
    Script for Section 2 figures
    :param no_figures: number of figures to generate
    """
    print('Generating figures for section 2...')
    for i in tqdm(range(no_figures)):
        mesh = morphable_model(FILE_MODEL)
        mesh_to_png(OUT / 'morphable_model/{:03d}.png'.format(i), mesh)


def section_3_1_figures():
    """
    Scrip for Section 2 Question 1 figures
    """
    mesh = file_to_mesh(FILE_MODEL)
    t1 = get_transformation_matrix(10)
    t2 = get_transformation_matrix(-10)
    mesh_r1 = transform_mesh(mesh, t1)
    mesh_r2 = transform_mesh(mesh, t2)
    mesh_to_png(OUT / 'rotation_10_degrees.png', mesh_r1)
    mesh_to_png(OUT / 'rotation_-10_degrees.png', mesh_r2)


def section_3_2_figure(rotation=0):
    """
    :param rotation: the rotation angle
    Script for Section 2 Question 2 figure
    """
    # open files
    mesh = file_to_mesh(FILE_MODEL)
    landmarks = np.loadtxt(FILE_LANDMARKS, dtype=np.int)
    # get the vertices from the landmarks
    mesh.vertices = mesh.vertices[landmarks]

    # get matrices
    transformation = get_transformation_matrix(rotation)
    viewport = get_viewport_matrix(width=WIDTH, height=HEIGHT)
    projection = get_projection_matrix(width=WIDTH, height=HEIGHT)

    # project mesh
    mesh_projected = project_mesh(mesh, transformation, viewport, projection)

    # plot
    points = mesh_projected.vertices[:, :2]
    plt.figure(figsize=(WIDTH, HEIGHT))
    plt.scatter(x=points[:, 0], y=points[:, 1], s=1)

    # annotate
    for i in range(len(landmarks)):
        plt.annotate(str(i), (points[i, 0], points[i, 1]))
    plt.savefig(OUT / 'landmarks_{:02d}_degrees.png'.format(rotation))


if __name__ == '__main__':
    plot_base()
    section_2_figures(10)
    section_3_1_figures()
    section_3_2_figure()
