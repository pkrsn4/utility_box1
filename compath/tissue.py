import numpy as np
import cv2

def extract_tissue_without_fat(img):
    
    BILATERAL1_ARGS={"d":9,"sigmaColor":10000,"sigmaSpace":150}
    BILATERAL2_ARGS={"d":90,"sigmaColor":5000,"sigmaSpace":5000}
    BILATERAL3_ARGS={"d":90,"sigmaColor":10000,"sigmaSpace":10000}
    BILATERAL4_ARGS={"d":90,"sigmaColor":10000,"sigmaSpace":100}
    
    THRESH1_ARGS={"thresh":0,"maxval":255,"type":cv2.THRESH_TRUNC+cv2.THRESH_OTSU}
    THRESH2_ARGS={"thresh":0,"maxval":255,"type":cv2.THRESH_OTSU}
    
    image = img.copy()
    w = img.shape[1]
    h = img.shape[0]
    
    shape=img.shape
    
    img_hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    
    lower_red=np.array([120,0,0])
    upper_red=np.array([180,255,255])
    
    mask=cv2.inRange(img_hsv,lower_red,upper_red)
    
    img_hsv=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2RGB)
    
    m=cv2.bitwise_and(img,img,mask=mask)
    
    im_fill=np.where(m==0,233,m)
    
    mask=np.zeros(shape[:2])
    
    gray=cv2.cvtColor(im_fill,cv2.COLOR_BGR2GRAY)
    blur1=cv2.bilateralFilter(np.bitwise_not(gray),**BILATERAL1_ARGS)
    blur2=cv2.bilateralFilter(np.bitwise_not(blur1),**BILATERAL2_ARGS)
    blur3=cv2.bilateralFilter(np.bitwise_not(blur2),**BILATERAL3_ARGS)
    blur4=cv2.bilateralFilter(np.bitwise_not(blur3),**BILATERAL4_ARGS)
    
    blur_final=255-blur4
    _,thresh=cv2.threshold(blur_final,**THRESH1_ARGS)
    _,thresh=cv2.threshold(thresh,**THRESH2_ARGS)
    
    return thresh


    
def extract_tissue(img):
    """
    """
    
    BILATERAL1_ARGS = {"d": 40, "sigmaColor": 100, "sigmaSpace": 150}
    THRESH1_ARGS = {"thresh": 0, "maxval": 255, "type": cv2.THRESH_TRUNC + cv2.THRESH_OTSU}
    THRESH2_ARGS = {"thresh": 0, "maxval": 255, "type": cv2.THRESH_OTSU}
    
    img = highPassFilter(img)
    img_temp = np.where(img < 100, 255, img)
    img_hsv = cv2.cvtColor(img_temp, cv2.COLOR_RGB2HSV)
    lower_red = np.array([100, 20, 0])
    upper_red = np.array([180, 255, 255])
    mask_red = cv2.inRange(img_hsv, lower_red, upper_red)
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_blue = np.array([0, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    
    img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    blue_patch = cv2.bitwise_and(img, img, mask=mask_blue)
    img = img - blue_patch
    
    m = cv2.bitwise_and(img, img, mask=mask_red)
    im_fill = np.where(m == 0, 233, m)
    
    image = img.copy()
    shape = img.shape
    gray = cv2.cvtColor(im_fill, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    blur_final = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    blur1 = cv2.bilateralFilter(blur_final, **BILATERAL1_ARGS)
    blur_final = cv2.bilateralFilter(blur1, **BILATERAL1_ARGS)
    blur_final = cv2.bilateralFilter(blur_final, **BILATERAL1_ARGS)
    
    _, thresh = cv2.threshold(blur_final, **THRESH1_ARGS)
    _, thresh = cv2.threshold(thresh, **THRESH2_ARGS)
    
    return(thresh)

def highPassFilter(img):
    
    BILATERAL1_ARGS = {"d": 40, "sigmaColor": 100, "sigmaSpace": 150}
    THRESH1_ARGS = {"thresh": 0, "maxval": 255, "type": cv2.THRESH_TRUNC + cv2.THRESH_OTSU}
    THRESH2_ARGS = {"thresh": 0, "maxval": 255, "type": cv2.THRESH_OTSU}
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_f = np.float32(img_gray)
    fur_img = cv2.dft(img_f, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(fur_img)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    rows, cols = img_f.shape
    crow, ccol = rows // 2, cols // 2
    fw = 200

    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[:, :ccol - fw, :] = 1
    mask[:, ccol + fw:, :] = 1
    mask[:crow - fw, ccol - fw:ccol + fw, :] = 1
    mask[crow + fw:, ccol - fw:ccol + fw, :] = 1
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = img_back * 255 // (np.max(img_back))
    img_back = img_back.astype(int)
    img_back = img_back.astype('uint8')
    blur = cv2.GaussianBlur(img_back, (5, 5), 0)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blur = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, ker)
    er_img = cv2.erode(blur, np.ones((5, 5), np.uint8))
    er_img = cv2.erode(er_img, np.ones((5, 5), np.uint8))
    blur = cv2.bilateralFilter(er_img, **BILATERAL1_ARGS)
    blur = cv2.bilateralFilter(blur, **BILATERAL1_ARGS)
    blur = cv2.bilateralFilter(blur, **BILATERAL1_ARGS)
    _, blur = cv2.threshold(blur, **THRESH1_ARGS)
    _, blur = cv2.threshold(blur, **THRESH2_ARGS)
    img = cv2.bitwise_and(img, img, mask=blur)
    return img