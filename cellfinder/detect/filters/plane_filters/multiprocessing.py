import numpy as np
from tifffile import tifffile
from sklearn.feature_extraction import image as sk_image
from cellfinder.detect.filters.plane_filters.classical_filter import (
    enhance_peaks,
)
from cellfinder.detect.filters.plane_filters.tile_walker import TileWalker


class MpTileProcessor(object):
    def __init__(self, thread_q, ball_filter_q):
        self.thread_q = thread_q
        self.ball_filter_q = ball_filter_q

    def process(
        self,
        plane_id,
        path,
        previous_lock,
        self_lock,
        clipping_value,
        threshold_value,
        soma_diameter,
        log_sigma_size,
        n_sds_above_mean_thresh,
    ):
        laplace_gaussian_sigma = log_sigma_size * soma_diameter
        plane = tifffile.imread(path)
        plane = plane.T
        np.clip(plane, 0, clipping_value, out=plane)

        walker = TileWalker(plane, soma_diameter, threshold_value)

        walker.walk_out_of_brain_only()

        thresholded_img = enhance_peaks(
            walker.thresholded_img,
            clipping_value,
            gaussian_sigma=laplace_gaussian_sigma,
        )

        # threshold
        plane = adaptive_thresholding(plane, 80, n_sds_above_mean_thresh, threshold_value)
        # avg = thresholded_img.ravel().mean()
        # sd = thresholded_img.ravel().std()
        #
        # plane[
        #     thresholded_img > avg + n_sds_above_mean_thresh * sd
        # ] = threshold_value
        tile_mask = walker.good_tiles_mask.astype(np.uint8)

        with previous_lock:
            pass
        self.ball_filter_q.put((plane_id, plane, tile_mask))
        self.thread_q.put(plane_id)
        self_lock.release()


def adaptive_thresholding(plane, wind, n_std, fill_value):
    patches = sk_image.extract_patches_2d(plane, (wind, wind))
    proc_patches = np.zeros(patches.shape)
    for n_patch in range(patches.shape[0]):
        proc_patches[n_patch, :, :] = meanstd_thr(patches[n_patch, :, :], n_std)
    img_back = sk_image.reconstruct_from_patches_2d(proc_patches, plane.shape)
    img_back = img_back > 0
    plane[img_back > 0] = fill_value
    return plane


def meanstd_thr(patch, n_std):
    avg = patch.ravel().mean()
    sd = patch.ravel().std()
    return patch > avg + n_std * sd