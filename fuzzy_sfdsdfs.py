import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[12]:


# Gaussian Function:
def gaussian(x, mean, std):
    return np.exp(-0.5*np.square((x-mean)/std))

# Membership Functions:
def ExtremelyDark(x, M):
    return gaussian(x, -50, M/6)

def VeryDark(x, M):
    return gaussian(x, 0, M/6)

def Dark(x, M):
    return gaussian(x, M/2, M/6)

def SlightlyDark(x, M):
    return gaussian(x, 5*M/6, M/6)

def SlightlyBright(x, M):
    return gaussian(x, M+(255-M)/6, (255-M)/6)

def Bright(x, M):
    return gaussian(x, M+(255-M)/2, (255-M)/6)

def VeryBright(x, M):
    return gaussian(x, 255, (255-M)/6)

def ExtremelyBright(x, M):
    return gaussian(x, 305, (255-M)/6)


def output_fuzzy_set(x, f, M, thres):
    x = np.array(x)
    result = f(x, M)
    result[result > thres] = thres
    return result

def AggregateFuzzySets(fuzzy_sets):
    return np.max(np.stack(fuzzy_sets), axis=0)

def Infer(i, M, get_fuzzy_set=False):
    # Calculate degree of membership for each class
    VD = VeryDark(i, M)
    Da = Dark(i, M)
    SD = SlightlyDark(i, M)
    SB = SlightlyBright(i, M)
    Br = Bright(i, M)
    VB = VeryBright(i, M)
    
    # Fuzzy Inference:
    x = np.arange(-50, 306)
    Inferences = (
        output_fuzzy_set(x, ExtremelyDark, M, VD),
        output_fuzzy_set(x, VeryDark, M, Da),
        output_fuzzy_set(x, Dark, M, SD),
        output_fuzzy_set(x, Bright, M, SB),
        output_fuzzy_set(x, VeryBright, M, Br),
        output_fuzzy_set(x, ExtremelyBright, M, VB)
    )
    
    # Calculate AggregatedFuzzySet:
    fuzzy_output = AggregateFuzzySets(Inferences)
    
    # Calculate crisp value of centroid
    if get_fuzzy_set:
        return np.average(x, weights=fuzzy_output), fuzzy_output
    return np.average(x, weights=fuzzy_output)


# Proposed fuzzy method
def fuzzy_contrast_enhance(rgb_image):
    # Convert RGB to LAB
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    
    # Get L channel
    l = lab[:, :, 0]
    
    # Calculate M value
    M = np.mean(l)
    if M < 128:
        M = 127 - (127 - M)/2
    else:
        M = 128 + M/2
        
    # Precompute the fuzzy transform
    x = list(range(-50,306))
    FuzzyTransform = dict(zip(x,[Infer(np.array([i]), M) for i in x]))
    
    # Apply the transform to l channel
    u, inv = np.unique(l, return_inverse = True)
    l = np.array([FuzzyTransform[i] for i in u])[inv].reshape(l.shape)
    
    # Min-max scale the output L channel to fit (0, 255):
    Min = np.min(l)
    Max = np.max(l)
    lab[:, :, 0] = (l - Min)/(Max - Min) * 255
    
    # Convert LAB to RGB
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# Traditional method of histogram equalization
def HE(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# Contrast Limited Adaptive Histogram Equalization
def CLAHE(rgb):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# # img = cv2.imread('LeNa_color.png')
# # fce = FuzzyContrastEnhance(img)


# import numpy as np
# from matplotlib.pyplot import imread
# from matplotlib.pyplot import imsave

# def triangular_membership_function(triangle_start, triangle_peak, triangle_end):
#     def membership_function(parameter):
#         if parameter < triangle_start:
#             return 0

#         if triangle_start <= parameter and parameter < triangle_peak:
#             return (parameter - triangle_start) / (triangle_peak - triangle_start)

#         if triangle_peak <= parameter and parameter < triangle_end:
#             return 1 - (parameter - triangle_peak) / (triangle_end - triangle_peak)

#         # triangle_end <= parameter
#         return 0

#     return membership_function


# def sigma_membership_function(sigma_start, sigma_end):
#     def membership_function(parameter):
#         if parameter < sigma_start:
#             return 0

#         if sigma_start <= parameter and parameter < sigma_end:
#             return (parameter - sigma_start) / (sigma_end - sigma_start)

#         # sigma_end <= parameter
#         return 1

#     return membership_function


# def inverse_sigma_membership_function(sigma_start, sigma_end):
#     def membership_function(parameter):
#         if parameter < sigma_start:
#             return 1

#         if sigma_start <= parameter and parameter < sigma_end:
#             return 1 - (parameter - sigma_start) / (sigma_end - sigma_start)

#         # sigma_end <= parameter
#         return 0

#     return membership_function


# def fuzzy_contrast_enhance(image):
    """
    Parameters
    ----------
    image : numpy array, gray scale
        image to be manipulated.
    """
    dark_color = 0
    gray_color = 127
    bright_color = 255

    # The membership parameters can be modified, if the result
    # doesn't satisfy your expectations
    gray_membership_function = np.vectorize(
        triangular_membership_function(65, gray_color, 190))
    bright_membership_function = np.vectorize(
        sigma_membership_function(gray_color, 145))
    dark_membership_function = np.vectorize(
        inverse_sigma_membership_function(80, gray_color))

    dark_image_part = dark_membership_function(image)
    gray_image_part = gray_membership_function(image)
    bright_image_part = bright_membership_function(image)

    enhanced_image = (dark_image_part * dark_color +
                      gray_image_part * gray_color +
                      bright_image_part * bright_color) / \
        (dark_image_part + gray_image_part + bright_image_part)

    enhanced_image = enhanced_image.astype(np.uint8)
    return enhanced_image
