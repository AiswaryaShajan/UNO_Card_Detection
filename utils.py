import cv2
import numpy as np

def apply_dilation(image, kernel_size=(3, 3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)
    return dilated_image

def resize_image(image, max_width=800, max_height=800):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized_image
    return image

def draw_color_label(card_image, color, best_match):
    if color and best_match:
        text = f"{color} {best_match}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (20, 50)
        font_scale = 1.2
        font_color = (255, 255, 255)
        outline_color = (0, 0, 0)
        thickness = 2
        outline_thickness = thickness + 2
        cv2.putText(card_image, text, position, font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
        cv2.putText(card_image, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)