from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, \
    SegChannelSelectionTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor

from .mirror_transform import MirrorTransform
from .custom_transforms import MaskTransform, ConvertSegmentationToRegionsTransform
from .default_data_augmentation import default_3D_augmentation_params
from .downsampling import DownsampleSegForDSTransform3, DownsampleSegForDSTransform2
from .pyramid_augmentations import MoveSegAsOneHotToData, ApplyRandomBinaryOperatorTransform, \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform

try:
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
except ImportError as ie:
    NonDetMultiThreadedAugmenter = None


def get_moreDA_augmentation(dataloader_train, dataloader_val, patch_size, params=default_3D_augmentation_params,
                            border_val_seg=-1,
                            seeds_train=None, seeds_val=None, order_seg=1, order_data=3, deep_supervision_scales=None,
                            soft_ds=False,
                            classes=None, pin_memory=True, regions=None,
                            use_nondetMultiThreadedAugmenter: bool = False):
    assert params.get('mirror') is None, "old version of params, use new keyword do_mirror"

    tr_transforms = []

    if params.get("selected_data_channels") is not None: # None
        tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

    # if params.get("selected_seg_channels") is not None: # [0]
    #     tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    patch_size_spatial = patch_size
    ignore_axes = None

    # patch_size_spatial = [112, 128, 112]
    # can handle seg channel > 1
    # tr_transforms.append(SpatialTransform(
    #     patch_size_spatial, patch_center_dist_from_border=None,
    #     do_elastic_deform=False, alpha=params.get("elastic_deform_alpha"),
    #     sigma=params.get("elastic_deform_sigma"),
    #     do_rotation=False, angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
    #     angle_z=params.get("rotation_z"), p_rot_per_axis=params.get("rotation_p_per_axis"),
    #     do_scale=False, scale=params.get("scale_range"),
    #     border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
    #     border_mode_seg="constant", border_cval_seg=border_val_seg,
    #     order_seg=order_seg, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
    #     p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
    #     independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    # ))

    # nothing to do with the seg
    # tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    # nothing to do with the seg
    # tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2, p_per_channel=0.5))
    # nothing to do with the seg
    # tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))

    # if params.get("do_additive_brightness"):
    #     tr_transforms.append(BrightnessTransform(params.get("additive_brightness_mu"),
    #                                              params.get("additive_brightness_sigma"),
    #                                              True, p_per_sample=params.get("additive_brightness_p_per_sample"),
    #                                              p_per_channel=params.get("additive_brightness_p_per_channel")))

    # nothing to do with the seg
    # tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))

    # nothing to do with the seg
    # tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
    #                                                     p_per_channel=0.5,
    #                                                     order_downsample=0, order_upsample=3, p_per_sample=0.25,
    #                                                     ignore_axes=ignore_axes))
    # nothing to do with the seg
    # tr_transforms.append(
    #     GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"),
    #                    p_per_sample=0.1))  # inverted gamma
    
    # choose this
    # nothing to do with the seg
    # if params.get("do_gamma"):
    #     tr_transforms.append(
    #         GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
    #                        p_per_sample=params["p_gamma"]))
    
    # if params.get("do_mirror") or params.get("mirror"): # choose this
    #     # can handle seg channel > 1
    #     tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    # if params.get("mask_was_used_for_normalization") is not None:
    #     mask_was_used_for_normalization = params.get("mask_was_used_for_normalization") # OrderedDict([(0, False)])
    #     tr_transforms.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0)) # it seems it doesn't do anything, need check
    
    # can handle seg channel > 1
    tr_transforms.append(RemoveLabelTransform(-1, 0))

    tr_transforms.append(RenameTransform('seg', 'target', True))

    # if regions is not None: # None not run
    #     tr_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    # if deep_supervision_scales is not None:
    #     if soft_ds:
    #         assert classes is not None
    #         tr_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
    #     else:
    #         # seg.shape (b, c, x, y, z)
    #         # choose this can handle seg channel > 1 
    #         tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
    #                                                           output_key='target'))

    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    tr_transforms = Compose(tr_transforms)

    if use_nondetMultiThreadedAugmenter:
        if NonDetMultiThreadedAugmenter is None:
            raise RuntimeError('NonDetMultiThreadedAugmenter is not yet available')
        batchgenerator_train = NonDetMultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                            params.get("num_cached_per_thread"), seeds=seeds_train,
                                                            pin_memory=pin_memory)
    else:
        batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                      params.get("num_cached_per_thread"),
                                                      seeds=seeds_train, pin_memory=pin_memory)
    # batchgenerator_train = SingleThreadedAugmenter(dataloader_train, tr_transforms)
    # import IPython;IPython.embed()

    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get("selected_data_channels") is not None:
        val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
    # if params.get("selected_seg_channels") is not None:
    #     val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    # params.get("move_last_seg_chanel_to_data") = False below code won't run
    # if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
    #     val_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))

    val_transforms.append(RenameTransform('seg', 'target', True))

    # if regions is not None:
    #     val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    # if deep_supervision_scales is not None:
    #     if soft_ds:
    #         assert classes is not None
    #         val_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
    #     else:
    #         val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
    #                                                            output_key='target'))

    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)

    if use_nondetMultiThreadedAugmenter:
        if NonDetMultiThreadedAugmenter is None:
            raise RuntimeError('NonDetMultiThreadedAugmenter is not yet available')
        batchgenerator_val = NonDetMultiThreadedAugmenter(dataloader_val, val_transforms,
                                                          max(params.get('num_threads') // 2, 1),
                                                          params.get("num_cached_per_thread"),
                                                          seeds=seeds_val, pin_memory=pin_memory)
    else:
        batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms,
                                                    max(params.get('num_threads') // 2, 1),
                                                    params.get("num_cached_per_thread"),
                                                    seeds=seeds_val, pin_memory=pin_memory)
    # batchgenerator_val = SingleThreadedAugmenter(dataloader_val, val_transforms)

    return batchgenerator_train, batchgenerator_val

