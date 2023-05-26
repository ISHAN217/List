import streamlit as st
from PIL import Image
from model import get_caption_model, generate_caption

def main():
    st.title("Image Captioning Web App")
    st.write("Upload an image and get AI-generated captions!")

    # Load pre-trained model
    model_weights_path = ("C:/Users/KIIT/Downloads/Listed/model.h5")  # Specify the path to your pre-trained model weights
    model = get_caption_model()

    # Load the model weights
    model.load_weights(model_weights_path)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process the image and generate captions
        captions = generate_caption(image, model)

        # Display the captions
        st.header("Generated Captions")
        for caption in captions:
            st.write("- " + caption)

if __name__ == '__main__':
    main()
