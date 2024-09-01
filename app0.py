import gradio as gr
import cv2
import torch
import numpy as np
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, HistogramNormalize,
    CropForeground, Zoom, Resize, DivisiblePad, ToTensor
)

from main_frustuminv_xray import NVLightningModule, make_cameras_dea

# Define the transformation pipeline for "image2d"
val_transforms = Compose(
    [
        # LoadImage(image_only=True),
        # EnsureChannelFirst(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        HistogramNormalize(min=0.0, max=1.0),
        CropForeground(select_fn=lambda x: x > 0, margin=0),
        Zoom(zoom=0.95, padding_mode="constant", mode="area"),
        Resize(spatial_size=256, size_mode="longest", mode="area"),
        DivisiblePad(k=(256, 256), mode="constant", constant_values=0),
        ToTensor()
    ]
)

def main():
    def image_to_image(image):
        # Run forward pass
        device = torch.device('cuda:0')
        B = 1

        # Ensure the channel is first
        if image.ndim == 2:  # Grayscale image
            image = np.expand_dims(image, axis=0)

        sample = val_transforms(image).to(device)
        sample = sample.unsqueeze(0)
        print(sample.shape)

        checkpoint_path = "logs/diffusion/version_3/checkpoints/copy.ckpt"
        model = NVLightningModule.load_from_checkpoint(checkpoint_path, strict=False).to(device)
        dist_hidden = 8 * torch.ones(B, device=device)
        elev_hidden = torch.zeros(B, device=device)
        azim_hidden = torch.zeros(B, device=device)
        view_hidden = make_cameras_dea(
            dist_hidden, 
            elev_hidden, 
            azim_hidden, 
            fov=16.0, 
            znear=6.5, 
            zfar=9.5
        ).to(device)

        # timesteps = torch.zeros((B,), device=device).long()\
        #     if timesteps is None else timesteps

        output = model.forward_volume(
            image2d=sample, 
            cameras=view_hidden, 
            noise=torch.zeros_like(sample),
        )

        # Convert to numpy
        output = output.to('cpu').detach().numpy().squeeze()
        print(output.shape)
        return output[128,:,:]
    
    # Load the Lena image
    # image = cv2.imread("data/xr/0a8dc3de200bc767169b73a6ab91e948.png")

    # Create Gradio interface
    iface = gr.Interface(
        fn=image_to_image,
        inputs=gr.Image(type="numpy", image_mode="L"),
        examples=[
            "data/ChestXRLungSegmentation/VinDr/v1/processed/test/images/0a8dc3de200bc767169b73a6ab91e948.png"
        ],
        outputs=gr.Image()
    )

    # Launch the app
    iface.launch()

if __name__ == "__main__":
    main()