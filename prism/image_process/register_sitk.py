from pathlib import Path
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from tifffile import imread, imwrite

import SimpleITK as sitk
import numpy as np

def register_3d_images(fixed_image_array, moving_image_array):
    # 将 numpy 数组转换为 SimpleITK 图像
    fixed_image = sitk.GetImageFromArray(fixed_image_array)
    moving_image = sitk.GetImageFromArray(moving_image_array)

    # **将图像转换为 Float32 类型**
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    # 初始化配准方法
    registration_method = sitk.ImageRegistrationMethod()

    # 使用互信息作为匹配指标
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    # 设置插值方法
    registration_method.SetInterpolator(sitk.sitkLinear)

    # 设置优化器
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-8
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # 使用中心化的初始变换
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # 执行配准
    final_transform = registration_method.Execute(fixed_image, moving_image)

    print("Optimizer's stopping condition: {0}".format(
        registration_method.GetOptimizerStopConditionDescription()))
    print("Final metric value: {0}".format(registration_method.GetMetricValue()))

    # 使用最终的变换对移动图像进行重采样
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(final_transform)
    resampled_moving_image = resampler.Execute(moving_image)

    # 将 SimpleITK 图像转换回 numpy 数组
    registered_image_array = sitk.GetArrayFromImage(resampled_moving_image)

    return registered_image_array, final_transform


def register_manual_sitk(ref_dir, src_dir, dest_dir, im_names=None):
    ref_dir = Path(ref_dir)
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)
    if im_names is None:
        im_names = [x.name for x in list(ref_dir.glob('*.tif'))]

    for im_name in tqdm(im_names, desc='Registering'):
        # 读取图像并转换为浮点类型
        ref_im = imread(ref_dir / im_name).astype(np.float32)
        src_im = imread(src_dir / im_name).astype(np.float32)

        # 调用 SimpleITK 进行配准
        out_im, _ = register_3d_images(ref_im, src_im)

        # 保存配准后的图像，注意类型转换
        imwrite(dest_dir / im_name, out_im.astype(np.uint16))

