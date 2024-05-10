import numpy as np
import matplotlib.pyplot as plt 

from skimage.color import rgb2gray
from skimage.feature import corner_harris, corner_peaks, BRIEF, match_descriptors
from skimage.transform import warp, ProjectiveTransform, SimilarityTransform 
from skimage.measure import ransac


def blending_alphas(img):
    alphas = np.zeros((img.shape[0] * img.shape[1]))
    max_dt = 0
    x, y = np.ogrid[:img.shape[0], :img.shape[1]]

    def get_alpha(i, j):
        alphas[i][j] = min([i, j, img.shape[0] - i - 1, img.shape[1] - j - 1])
        max_dt = max(max_dt, alphas[i][j])
    
    np.vectorize(get_alpha)(x, y)

    return alphas / max_dt


def create_panorama(imgs):
    base_img = imgs[0]
    base_img_grey = rgb2gray(imgs[0])
    base_peaks = corner_peaks(corner_harris(base_img_grey), threshold_rel=0.0005, min_distance=5)

    total_length = base_img.shape[1]
    current_right_translation = 0

    for img in imgs[1:]:
        img_gray = rgb2gray(img)
        img_peaks = corner_peaks(corner_harris(img_gray), threshold_rel=0.0005, min_distance=5)

        extractor = BRIEF()
        
        extractor.extract(base_img_grey, base_peaks)
        base_key_points = base_peaks[extractor.mask]
        base_descriptors = extractor.descriptors
        
        extractor.extract(img_gray, img_peaks)
        img_key_points = img_peaks[extractor.mask]
        img_descriptors = extractor.descriptors

        desc_matches = match_descriptors(base_descriptors, img_descriptors, cross_check=True)

        base_inliers = base_key_points[desc_matches.transpose()[0]]
        img_inliers = img_key_points[desc_matches.transpose()[1]]
        
        _, inliers = ransac(
            (base_inliers, img_inliers), 
            ProjectiveTransform, 
            min_samples=4, 
            residual_threshold=3, 
            max_trials=1000
        )
        key_points_base_result = base_inliers[inliers]
        key_points_img_result = img_inliers[inliers]

        num_inliers = key_points_img_result.shape[0]
        add_points = np.array([np.zeros(num_inliers), np.ones(num_inliers) * base_img.shape[1]]).transpose()

        transform = ProjectiveTransform()
        transform.estimate((key_points_img_result + add_points)[:, ::-1], key_points_base_result[:, ::-1])

        total_length += img.shape[1]
        current_right_translation += img.shape[1]

        stretch_translation = SimilarityTransform(translation=0)
        translation_img = SimilarityTransform(translation=[current_right_translation, 0])

        img_warped = warp(warp(img, translation_img.inverse, output_shape=(img.shape[0], total_length)), transform.inverse)

        base_alphas_warped = warp(blending_alphas(base_img), stretch_translation.inverse, output_shape=(img.shape[0], total_length))
        img_alphas_warped = warp(blending_alphas(img), translation_img.inverse, output_shape=(img.shape[0], total_length))
        img_alphas_warped = warp(img_alphas_warped, transform.inverse)

        base_img = warp(base_img, stretch_translation.inverse, output_shape=(base_img.shape[0], total_length))

        base_img_final = np.dstack((base_img, base_alphas_warped / (base_alphas_warped + img_alphas_warped)))
        img_final = np.dstack((img_warped, img_alphas_warped / (base_alphas_warped + img_alphas_warped)))

        base_img = (base_img_final + img_final if base_img.shape[2] < 4 else base_img + img_final) / 2
        base_img_grey = rgb2gray(base_img[:, :, :3])
        base_peaks = corner_peaks(corner_harris(base_img_grey), threshold_rel=0.0005, min_distance=5)
    
    return base_img


if __name__ == "__main__":
    result = create_panorama([plt.imread(f"images/t1/{i}.JPG") for i in range(1, 5)])

    plt.figure(figsize=(20, 16))
    plt.imshow(result[:, :5000, :])
    plt.show()