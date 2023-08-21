# Import Streamlit
import streamlit as st
# Import app specific functions
from constants import *
from utils import *

def main():
    st.title("Style Transfer using GAN")
    st.write(":blue[This Capstone project will translate T1 MRI image to T2 MRI image and vice-versa.]")
    st.write(":green[Developed By:] :violet[Sanjeev Surendran]")
    st.write("License : :red[This project is not a open source and sharing the project files is prohibited!]")
    uploaded_file = st.file_uploader("Choose an MRI image to convert", type="png")
    tab1, tab2, tab3 = st.tabs(["Input Image", "T1 Image", "T2 Image"])

    if uploaded_file is not None:
        # Load image
        image = load_img(uploaded_file)
        
        with tab1:
            # Display image with width 512
            st.image(image, caption="Uploaded Image", width=512)
            t1_image, t2_image = generate_image(uploaded_file)
        with tab2:
            # Display image with width 512
            st.image(t1_image, caption="T1 Image", width=512)
        with tab3:
            # Display image with width 512
            st.image(t2_image, caption="T2 Image", width=512)

if __name__ == "__main__":
    main()