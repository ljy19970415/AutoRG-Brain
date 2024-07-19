"""
Reference:
A self-supervised learning strategy for postoperative brain cavity segmentation simulating resections
""" 
import warnings
import numpy as np
from tqdm import tqdm
from noise import pnoise3, snoise3
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtk.numpy_interface import dataset_adapter as dsa



def add_noise_to_sphere(poly_data, octaves, offset=0, scale=0.5):
    """
    Expects sphere with radius 1 centered at the origin
    """
    wrap_data_object = dsa.WrapDataObject(poly_data)
    points = wrap_data_object.Points
    normals = wrap_data_object.PointData['Normals']

    points_with_noise = []
    zipped = list(zip(points, normals))
    for point, normal in zipped:#tqdm(zipped):
        offset_point = point + offset
        noise = scale * snoise3(*offset_point, octaves=octaves)
        point_with_noise = point + noise * normal
        points_with_noise.append(point_with_noise)
    points_with_noise = np.array(points_with_noise)

    vertices = vtk.vtkPoints()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        points_with_noise_vtk = numpy_to_vtk(points_with_noise)
    vertices.SetData(points_with_noise_vtk)
    poly_data.SetPoints(vertices)

    return poly_data


def center_poly_data(poly_data):
    centerOfMassFilter = vtk.vtkCenterOfMass()
    centerOfMassFilter.SetInputData(poly_data)
    centerOfMassFilter.SetUseScalarsAsWeights(False)
    centerOfMassFilter.Update()
    center = np.array(centerOfMassFilter.GetCenter())

    transform = vtk.vtkTransform()
    transform.Translate(-center)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(poly_data)
    transform_filter.Update()

    poly_data = transform_filter.GetOutput()
    return poly_data


def transform_poly_data(poly_data, center, radii, angles):
    transform = vtk.vtkTransform()
    transform.Translate(center)
    # x_angle, y_angle, z_angle = angles  # there must be a better way
    # transform.RotateX(x_angle)
    # transform.RotateY(y_angle)
    # transform.RotateZ(z_angle)
    transform.Scale(*radii)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(poly_data)
    transform_filter.Update()

    poly_data = transform_filter.GetOutput()
    return poly_data


def compute_normals(poly_data):
    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.AutoOrientNormalsOn()
    normal_filter.SetComputePointNormals(True)
    normal_filter.SetComputeCellNormals(True)
    normal_filter.SplittingOff()
    normal_filter.SetInputData(poly_data)
    normal_filter.ConsistencyOn()
    normal_filter.Update()
    poly_data = normal_filter.GetOutput()
    return poly_data


def get_resection_poly_data(
        poly_data,
        offset,
        center,
        radii,
        angles,
        octaves=4,
        scale=0.5,
        deepcopy=True,
        ):
    if deepcopy:
        new_poly_data = vtk.vtkPolyData()
        new_poly_data.DeepCopy(poly_data)
        poly_data = new_poly_data

    poly_data = add_noise_to_sphere(poly_data, octaves=octaves, offset=offset)
    poly_data = center_poly_data(poly_data)

    poly_data = transform_poly_data(poly_data, center, radii, angles)
    poly_data = compute_normals(poly_data)
    return poly_data
