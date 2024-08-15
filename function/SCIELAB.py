import numpy as np
import cv2
from scipy.io import loadmat
from scipy.ndimage import convolve1d
from skimage import color

def load_display_calibration(mat_dir):
    display_spd = loadmat(f'{mat_dir}/displaySPD.mat')['displaySPD']
    cones = loadmat(f'{mat_dir}/SmithPokornyCones.mat')['cones']
    gamma_table = loadmat(f'{mat_dir}/displayGamma.mat')['invGamma']
    rgb2lms = np.dot(cones.T, display_spd)
    return rgb2lms, gamma_table

def dac2rgb(DAC, GammaTable=2.2):
    if isinstance(GammaTable, (int, float)):
        print(f'Raising DAC values to a power of {GammaTable}')
        RGB = DAC ** GammaTable
    elif isinstance(GammaTable, (list, tuple, np.ndarray)) and len(GammaTable) == 3:
        print(f'Raising R values to a power of {GammaTable[0]}')
        print(f'Raising G values to a power of {GammaTable[1]}')
        print(f'Raising B values to a power of {GammaTable[2]}')
        RGB = np.empty_like(DAC)
        RGB[..., 0] = DAC[..., 0] ** GammaTable[0]
        RGB[..., 1] = DAC[..., 1] ** GammaTable[1]
        RGB[..., 2] = DAC[..., 2] ** GammaTable[2]
    else:
        DAC = np.round(DAC * (GammaTable.shape[0] - 1)).astype(int)
        if GammaTable.shape[1] == 1:
            RGB = GammaTable[DAC]
        else:
            RGB = np.empty_like(DAC, dtype=float)
            RGB[..., 0] = GammaTable[DAC[..., 0], 0]
            RGB[..., 1] = GammaTable[DAC[..., 1], 1]
            RGB[..., 2] = GammaTable[DAC[..., 2], 2]
    return RGB

def change_color_space(image, transform_matrix):
    return np.dot(image, transform_matrix.T)

def get_planes(image):
    return image[..., 0], image[..., 1], image[..., 2]

def pad_for_conv(image, filter_length):
    pad_width = filter_length // 2
    return np.pad(image, pad_width, mode='reflect')

def resize(image, shape):
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

def separable_filters(samp_per_deg, dimension):
    sigma1 = 0.05 * samp_per_deg
    sigma2 = 0.15 * samp_per_deg
    sigma3 = 0.40 * samp_per_deg
    size = int(np.ceil(sigma3 * 6))

    k1 = np.exp(-0.5 * (np.linspace(-size, size, 2 * size + 1) / sigma1) ** 2)
    k2 = np.exp(-0.5 * (np.linspace(-size, size, 2 * size + 1) / sigma2) ** 2)
    k3 = np.exp(-0.5 * (np.linspace(-size, size, 2 * size + 1) / sigma3) ** 2)

    return k1 / k1.sum(), k2 / k2.sum(), k3 / k3.sum()

def separable_conv(image, k1, k2):
    temp = convolve1d(image, k1, axis=0)
    return convolve1d(temp, k2, axis=1)

def scielab(samp_per_deg, image1, image2, whitepoint, imageformat, k=None):
    if imageformat == 'xyz10' or imageformat == 'lms10':
        xyztype = 10
    else:
        xyztype = 2

    if imageformat.startswith('lms'):
        opp1 = change_color_space(image1, cmatrix('lms2opp'))
        opp2 = change_color_space(image2, cmatrix('lms2opp'))
        oppwhite = change_color_space(whitepoint, cmatrix('lms2opp'))
        whitepoint = change_color_space(oppwhite, cmatrix('opp2xyz', xyztype))
    else:
        opp1 = change_color_space(image1, cmatrix('xyz2opp', xyztype))
        opp2 = change_color_space(image2, cmatrix('xyz2opp', xyztype))

    imsize = image1.shape
    w1, w2, w3 = get_planes(opp1)

    k1, k2, k3 = separable_filters(samp_per_deg, 3)

    w1 = pad_for_conv(w1, len(k1))
    w2 = pad_for_conv(w2, len(k2))
    w3 = pad_for_conv(w3, len(k3))

    p1 = separable_conv(w1, k1, np.abs(k1))
    p2 = separable_conv(w2, k2, np.abs(k2))
    p3 = separable_conv(w3, k3, np.abs(k3))

    new1 = np.stack((p1, p2, p3), axis=-1)

    w1, w2, w3 = get_planes(opp2)

    w1 = pad_for_conv(w1, len(k1))
    w2 = pad_for_conv(w2, len(k2))
    w3 = pad_for_conv(w3, len(k3))

    p1 = separable_conv(w1, k1, np.abs(k1))
    p2 = separable_conv(w2, k2, np.abs(k2))
    p3 = separable_conv(w3, k3, np.abs(k3))

    new2 = np.stack((p1, p2, p3), axis=-1)

    result = change_color_space(new1, cmatrix('opp2xyz', xyztype))
    result2 = change_color_space(new2, cmatrix('opp2xyz', xyztype))

    delta_e = delta_lab(result, result2, whitepoint)

    return delta_e

def cmatrix(transform_type, xyztype=2):
    if transform_type == 'lms2opp':
        return np.array([[0.4002, 0.7075, -0.0808],
                         [-0.2263, 1.1653, 0.0457],
                         [0, 0, 0.9182]])
    elif transform_type == 'xyz2opp':
        return np.array([[0.4002, 0.7075, -0.0808],
                         [-0.2263, 1.1653, 0.0457],
                         [0, 0, 0.9182]])
    elif transform_type == 'opp2xyz':
        return np.linalg.inv(cmatrix('xyz2opp'))

def delta_lab(result, result2, whitepoint):
    # Normalize whitepoint to [0, 1] range as expected by skimage's xyz2lab
    normalized_whitepoint = whitepoint / np.max(whitepoint)
    illuminant = 'D65'  # Use a standard illuminant
    lab1 = color.xyz2lab(result, illuminant=illuminant)
    lab2 = color.xyz2lab(result2, illuminant=illuminant)
    delta_e = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))
    return delta_e

def compute_scielab(image_path1, image_path2, bounder, mat_dir):
    def load_image(image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

    def crop_image(image, bounder):
        return image[bounder:-bounder, bounder:-bounder, :]

    image1 = load_image(image_path1)
    image2 = load_image(image_path2)

    if image1.shape != image2.shape:
        raise ValueError('The image size should be the same!')

    image1 = crop_image(image1, bounder)
    image2 = crop_image(image2, bounder)

    rgb2lms, gamma_table = load_display_calibration(mat_dir)

    img1_dac = dac2rgb(image1, gamma_table)
    lms1 = change_color_space(img1_dac, rgb2lms)

    img2_dac = dac2rgb(image2, gamma_table)
    lms2 = change_color_space(img2_dac, rgb2lms)

    samp_per_deg = 23
    white_point = np.dot(np.array([1, 1, 1]), rgb2lms.T)
    error_image = scielab(samp_per_deg, lms1, lms2, white_point, 'lms')
    scielab_error = np.mean(error_image)

    return scielab_error

image_path1 = r'/home/cnu_cdx/mamba_demosaick/NeWCRFs-master/datasets/nyu/scielab/hats.tiff'
image_path2 = r'/home/cnu_cdx/mamba_demosaick/NeWCRFs-master/datasets/nyu/scielab/hatsCompressed.tiff'
mat_dir = r'/home/cnu_cdx/mamba_demosaick/NeWCRFs-master/datasets/nyu/scielab'
bounder = 25

scielab_error = compute_scielab(image_path1, image_path2, bounder, mat_dir)
print(f"SCIELAB Error: {scielab_error}")
