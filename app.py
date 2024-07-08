# Author: Maciej Ołdakowski
import streamlit as st
import os
import torch
import numpy as np
from numpy.random import default_rng
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage
import torch.nn as nn

# Generator model architecture
latent_size = 128  # Assuming your latent size is 100
generator = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
)

# Model weights path
MODEL_PATH = 'models/G.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to generate image grid
def gen_image_grid(generator, x=5, y=5):
    latent_points = default_rng().normal(0.0, 1.0, (x*y, latent_size, 1, 1)).astype(np.float32)
    latent_points = torch.tensor(latent_points).to(device)
    with torch.no_grad():
        generator.eval()
        X = generator(latent_points).cpu()
    grid = make_grid(X, nrow=x, normalize=True, value_range=(-1, 1))
    return ToPILImage()(grid)

def main():
    if 'button' not in st.session_state:
        st.session_state['button'] = False

    overview = st.container()
    images = st.container()
    
    with overview:
        st.title("Generate your grid of abstract paintings")
        st.write("### With the help of GAN")
        st.write("Using a predefined generator model")

        x = st.number_input("Insert a x size of an image grid:",
                            min_value=7, max_value=20, value=10)
        y = st.number_input("Insert a y size of an image grid:",
                            min_value=7, max_value=20, value=10)
        if not isinstance(x, int) or not isinstance(y, int):
            st.error("x and y have to be an integer between 1 and 10!",
                     icon="🚨")
        st.session_state['button'] = st.button("Generate!")

    with images:
        if st.session_state['button']:
            # Load model checkpoint
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            generator.load_state_dict(checkpoint)

            # Generate images
            generator.eval()
            image = gen_image_grid(generator, x, y)
            st.image(image, caption='Generated Abstract Art', use_column_width=True)

# Run main program
if __name__ == '__main__':
    main()
