import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import numpy as np

def pd_to_numpy_vol(pd, spacing=[1.,1.,1.], shape=None, origin=None, foreground_value=255, backgroud_value = 0):
    if shape is None:
        bnds = np.array(pd.GetBounds())
        shape = np.ceil((bnds[1::2]-bnds[::2])/spacing).astype(int)+15
    if origin is None:
        origin = bnds[::2]+(bnds[1::2]-bnds[::2])/2

    #make image
    extent = np.zeros(6).astype(int)
    extent[1::2] = np.array(shape)-1
    
    imgvtk = vtk.vtkImageData()
    imgvtk.SetSpacing(spacing)
    imgvtk.SetOrigin(origin)
    imgvtk.SetExtent(extent)
    imgvtk.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    
    vtk_data_array = numpy_support.numpy_to_vtk(num_array=np.ones(shape[::-1]).ravel()*backgroud_value,  # ndarray contains the fitting result from the points. It is a 3D array
                                                deep=True, array_type=vtk.VTK_FLOAT)
    
    imgvtk.GetPointData().SetScalars(vtk_data_array)
    
    #poly2 stencil
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(pd)
    pol2stenc.SetOutputSpacing(spacing)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputWholeExtent(extent)
    pol2stenc.Update()
    
    #stencil to image
    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(imgvtk)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOn()
    imgstenc.SetBackgroundValue(foreground_value)
    imgstenc.Update()
    
    ndseg = numpy_support.vtk_to_numpy(imgstenc.GetOutputDataObject(0).GetPointData().GetArray(0))
    return ndseg.reshape(shape[::-1])


def pd_to_itk_image(pd, ref_img, foreground_value=255, backgroud_value = 0):
    ndseg = pd_to_numpy_vol(pd, spacing=ref_img.GetSpacing(), shape=ref_img.GetSize(), origin=ref_img.GetOrigin() )
    segitk = sitk.GetImageFromArray(ndseg.astype(np.int16))
    segitk.CopyInformation(ref_img)
    return segitk, ndseg



def rotate_img(img, rotation_center=None, theta_x=0,theta_y=0, theta_z=0, translation=(0,0,0), interp=sitk.sitkLinear, pixel_type=None, default_value=None):
    if not rotation_center:
        rotation_center = np.array(img.GetOrigin())+np.array(img.GetSpacing())*np.array(img.GetSize())/2
    if default_value is None:
        default_value = img.GetPixel(0,0,0)
    pixel_type = img.GetPixelIDValue()

    rigid_euler = sitk.Euler3DTransform(rotation_center, theta_x, theta_y, theta_z, translation)
    return sitk.Resample(img, img, rigid_euler, interp, default_value, pixel_type)

def rotate_polydata(pd, rotation_center, theta_x=0,theta_y=0, theta_z=0, translation=(0,0,0)):
    rigid_euler = sitk.Euler3DTransform(rotation_center, -theta_x, -theta_y, -theta_z, translation)
    matrix = np.zeros([4,4])
    old_matrix=np.array(rigid_euler.GetMatrix()).reshape(3,3)
    matrix[:3,:3] = old_matrix
    matrix[-1,-1] = 1

    # to rotate about a center we first need to translate
    transform_t = vtk.vtkTransform()
    transform_t.Translate(-rotation_center)
    transformer_t = vtk.vtkTransformPolyDataFilter()
    transformer_t.SetTransform(transform_t)
    transformer_t.SetInputData(pd)
    transformer_t.Update()

    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix.ravel())

    transformer = vtk.vtkTransformPolyDataFilter()
    transformer.SetTransform(transform)
    transformer.SetInputConnection(transformer_t.GetOutputPort())
    transformer.Update()

    # translate back
    transform_t2 = vtk.vtkTransform()
    transform_t2.Translate(rotation_center)
    transformer_t2 = vtk.vtkTransformPolyDataFilter()
    transformer_t2.SetTransform(transform_t2)
    transformer_t2.SetInputConnection(transformer.GetOutputPort())
    transformer_t2.Update()

    return transformer_t2.GetOutputDataObject(0)
