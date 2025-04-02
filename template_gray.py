import cv2 as cv
import os

def convert_templates_to_gray(template_folder, save_folder):
    """Converts all images in the template folder to grayscale and saves them."""
    # Create a save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Get all image files from the template folder
    template_files = [f for f in os.listdir(template_folder) if f.endswith('.jpg')]

    # Process each template
    for template_file in template_files:
        # Read the image
        template_path = os.path.join(template_folder, template_file)
        template_image = cv.imread(template_path)

        if template_image is not None:
            # Convert the image to grayscale
            gray_template = cv.cvtColor(template_image, cv.COLOR_BGR2GRAY)

            # Save the grayscale image in the save folder
            gray_template_path = os.path.join(save_folder, f"gray_{template_file}")
            cv.imwrite(gray_template_path, gray_template)
            print(f"Saved grayscale template: {gray_template_path}")
        else:
            print(f"Could not read template: {template_file}")

# Specify the paths
template_folder = 'templates'  # Update this path
save_folder = 'save'  # Update this path

# Call the function to convert templates to grayscale
convert_templates_to_gray(template_folder, save_folder)
