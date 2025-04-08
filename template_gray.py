import cv2 as cv
import os

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))

    return cv.resize(image, dim, interpolation=inter)

def convert_templates_to_binary(template_folder, save_folder, threshold_value=128):
    """Converts templates to binary (digit black, background white) while keeping aspect ratio."""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    template_files = [f for f in os.listdir(template_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for template_file in template_files:
        template_path = os.path.join(template_folder, template_file)
        template_image = cv.imread(template_path)

        if template_image is not None:
            original_shape = template_image.shape[:2]

            # Upscale for cleaner thresholding
            upscale = cv.resize(template_image, None, fx=2.0, fy=2.0, interpolation=cv.INTER_CUBIC)

            # Convert to grayscale
            gray = cv.cvtColor(upscale, cv.COLOR_BGR2GRAY)

            # Binary thresholding (Digit becomes BLACK, BG becomes WHITE)
            _, binary = cv.threshold(gray, threshold_value, 255, cv.THRESH_BINARY)

            # Resize back while keeping aspect ratio
            binary_resized = resize_with_aspect_ratio(binary, width=original_shape[1])

            # Save
            binary_path = os.path.join(save_folder, f"binary_{template_file}")
            cv.imwrite(binary_path, binary_resized)
            print(f"Saved binary template: {binary_path}")
        else:
            print(f"Could not read template: {template_file}")

# Set your paths
template_folder = 'templates'
save_folder = 'processed_templates'

convert_templates_to_binary(template_folder, save_folder)
