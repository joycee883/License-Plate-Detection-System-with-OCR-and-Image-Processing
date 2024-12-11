import streamlit as st 
import numpy as np
import cv2
import imutils # To resize images
import pytesseract # To extract text from the license plate
from PIL import Image
import base64

# Function to read and encode the image file
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Set the background image using CSS
def set_background(image_base64):
    page_bg_img = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
    }}
    .css-1g8v9l0 {{
        background: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
    }}
    .stButton > button {{
        background-color: #4C4C6D;
        color: white;
    }}
    .stButton > button:hover {{
        background-color: #6A5ACD;
        color: white;
    }}
    .stSlider > div {{
        background-color: transparent;
    }}
    .stSelectbox div {{
        color: white;
    }}
    .stSubheader {{
        color: white;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Call the function with the uploaded background image
image_base64 = get_base64_image("bg2.jpeg")  # Path to your uploaded image
set_background(image_base64)


# Set up pytesseract command (Update the path to your tesseract executable)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\91939\Desktop\AI&DS\Data science projects\Number Plate Detection'

# Page Title
st.title("🚘 Intelligent Vehicle License Plate Detection🔍")
st.subheader("✨Description")
st.info("🔎 What is this project about?")
st.markdown(
    """
    <div style="background-color:#0E1F44; padding:10px; border-radius:5px;">
        <p style="color:white; text-align:justify;">
        This project focuses on automatically detecting and extracting license plates from vehicle images using advanced image processing techniques. The goal is to create an efficient system that isolates license plates for further analysis, paving the way for intelligent traffic management and smart city solutions.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Upload Image
input = st.file_uploader("📂 Upload a Vehicle Image", type=['jpg', 'jpeg', 'png'])

if st.button("Click Here"):
    if input is not None:
        
        # 🖼️ Display Uploaded Image
        image = Image.open(input)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert image to NumPy array for OpenCV
        image = np.array(image)
        
        # 🔄 Preprocessing
        st.subheader("🔄 Image Preprocessing")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
        st.image(gray_image, caption="Grayscale and Smoothened Image", use_column_width=True, channels="GRAY")
        
        # 🔍 Edge Detection
        edged = cv2.Canny(gray_image, 30, 200)
        st.image(edged, caption="Edges Detected", use_column_width=True, channels="GRAY")

        # 🔳 Contour Detection
        st.subheader("🔳 Contour Detection")
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
        
        screenCnt = None
        new_img = None
        for c in cnts:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
            if len(approx) == 4:
                screenCnt = approx
                x, y, w, h = cv2.boundingRect(c)
                new_img = image[y:y + h, x:x + w]
                break

        # Check if license plate was detected
        if screenCnt is not None:
            # 🖼️ Display Detected Plate
            cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
            st.image(image, caption="Detected License Plate", use_column_width=True)

            # Extract and Display Cropped License Plate
            st.subheader("📋 Extracted License Plate")
            st.image(new_img, caption="Cropped License Plate", use_column_width=True)

            # 🖨️ Provide Download Option for Cropped License Plate
            if new_img is not None:
                new_img_pil = Image.fromarray(new_img)
                new_img_pil.save("cropped_license_plate.png")  # Save locally
                with open("cropped_license_plate.png", "rb") as file:
                    st.download_button(
                        label="💾 Download Extracted License Plate",
                        data=file,
                        file_name="cropped_license_plate.png",
                        mime="image/png",
                    )
        else:
            st.error("⚠️ Could not detect a license plate. Try uploading a clearer image.")
else:
    st.info("📥 Please upload a vehicle image to begin.")
    
    
# Sidebar About Section
st.sidebar.title("🚗 About This App")
st.sidebar.write(
    """
    Welcome to the **Smart Plate Detector**!  

    🛠️ **What it does:**  
    Detects and isolates license plates from uploaded images in a few simple steps.  

    🚦 **Why it’s useful:**  
    A crucial tool for modern traffic management, parking systems, and vehicle tracking.  

    📋 **Quick Tip:**  
    Use clear, high-quality images for the best results.  

    Let's make road intelligence simple and efficient! 🚘  
    """
)

