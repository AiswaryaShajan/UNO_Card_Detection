# UNO Card Detection 🟥🟨🟩🟦

A Python-based project for detecting and recognizing UNO cards in real-time using computer vision techniques.

## Features

- Detects UNO cards from images or webcam.
- Recognizes card number/symbol using SIFT-based template matching.
- Identifies dominant card color using HSV masking.
- Includes perspective transform for accurate preprocessing.
- Modular codebase (each task handled in separate scripts).

## Project Structure

```
   ├──templates/ → Preprocessed UNO card templates  
   ├──main.py → CLI entry point to run webcam or image detection
   ├──app_ui.py → Tkinter-based GUI for user-friendly interface
   ├──ui.py → Handles user input, image/webcam routing
   ├──feature_matching.py → SIFT + BFMatcher for card feature detection 
   ├──colour_detection.py → Detects card color using HSV masks
   ├──card_detection.py → Finds card contours and applies perspective corrections
   ├──utils.py → Helper utilities (resizing, dilation, overlay text)
   ```

---

## Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/AiswaryaShajan/UNO_Card_Detection.git
   cd UNO_Card_Detection
2. Install Dependencies
   ```bash
   pip install -r requirements.txt
3. Running the Program
   ```bash
   python3 app_ui.py
   ```
    You’ll be prompted to choose between webcam or image-based detection.

## How It Works
- 🃏 Card Detection
    - What happens: The input image or webcam frame is scanned for contours. The largest quadrilateral contour is assumed to be the card.

    - Output: A transformed (flattened) image of the card is generated using perspective transform.
      

- 🌈 Color Detection (HSV Masking)
    - What happens: The transformed card image is converted to HSV. Color masks for red, green, blue, and yellow are applied.

    - Output: The color with the highest mask area is selected as the dominant card color.


- Feature Matching
    - SIFT keypoints are extracted from the card and matched against the stored templates using BFMatcher with Lowe's ratio test.

- Label Overlay
    - The detected color and card name (e.g., "Red 4") are drawn on the output frame.

- Output
     <img src="images/card_detection.JPG" alt="card_detection" width="500"/>

## Areas of Improvement

1. **Performance**:  
   - The program can be **slow** at times, especially during processing. Exploring optimizations in the code or using hardware acceleration (e.g., GPU) could help improve speed and responsiveness.

2. **Accuracy in Number Recognition**:  
   - There are occasional mix-ups with certain numbers, such as **`6` and `9` or `1` and `7`**. Refining the feature matching or improving the template designs could help reduce these errors.

3. **Color Detection**:  
   - The system sometimes struggles to distinguish between colors like **green and yellow**. Adjusting the HSV thresholds or incorporating more advanced color classification methods could enhance reliability.

4. **Background Dependency**:  
   - The program performs best when cards are placed against a dark background. Adding better background segmentation or preprocessing techniques might make it more adaptable to different environments.

## What Next? 🔍:  
   - Train a machine learning model for classification the cards for better detection accuracy and robust system.



