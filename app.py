import gradio as gr
import cv2
import torch
import torchvision
from tqdm import tqdm
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

def generate_rotating_volume(model, volume, n_frames = 256, device=None):
    B = 1
    volume = torch.permute(volume, (0, 1, 3, 4, 2))
    frames = []
    print('Generating rotating volume ...')
    for az in tqdm(range(n_frames)):
        dist = 8 * torch.ones(B, device=device)
        elev = torch.zeros(B, device=device)
        azim = (az / n_frames * 2) * torch.ones(B, device=device)
        cameras = make_cameras_dea(
            dist, 
            elev, 
            azim, 
            fov=16.0, 
            znear=6.5, 
            zfar=9.5
        ).to(device)
        screen = model.forward_screen(
            image3d=volume, 
            cameras=cameras, 
        ).clamp_(0, 1).squeeze().detach().cpu() #.numpy()
        # screen = screen[:, ::-1]
        screen = torch.flip(screen, (1,))
        screen = screen.transpose(1, 0)
        frames.append(screen)
        # frames.append(volume_model(camera)[..., :3].clamp(0.0, 1.0))
    frames = torch.from_numpy(np.array(frames))
    return frames

def main():
    def image_to_volume(image):
        # Run forward pass
        device = torch.device('cuda:0')
        B = 1

        # Ensure the channel is first
        if image.ndim == 2:  # Grayscale image
            image = np.expand_dims(image, axis=0)

        sample = val_transforms(image).to(device)
        sample = sample.unsqueeze(0)
        # print(sample.shape)

        checkpoint_path = "logs/diffusion/version_3/checkpoints/last.ckpt"
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

        output = model.forward_volume(
            image2d=sample, 
            cameras=view_hidden, 
            noise=torch.zeros_like(sample),
        )
    
        rotates = generate_rotating_volume(
            model=model, 
            volume=output,
            n_frames=256,
            device=device, 
        )
        print(rotates.shape)
        
        outputs = output.clamp_(0, 1).squeeze().detach().cpu() #.numpy()

        # Determine the midpoint
        midpoint = len(rotates) // 4
        rotates = torch.cat([rotates[midpoint:,:,:], rotates[:midpoint,:,:]], dim=0)
    
        # Convert the input image to a video
        height, width = (256, 256)
        video_name = '360view.mp4'
        outputs = 255*outputs.unsqueeze(-1).repeat(1,1,1,3)
        rotates = 255*rotates.unsqueeze(-1).repeat(1,1,1,3)
        print(outputs.shape)
        print(rotates.shape)
        
        torchvision.io.write_video(
            filename = video_name,
            video_array = rotates,
            fps = 32.0
        )

        # video_file = cv2.VideoWriter(
        #     video_name, 
        #     cv2.VideoWriter_fourcc(*'H264'), 
        #     1, 
        #     (width, height)
        # )

        # # Create a video with n frames of the same image
        # for t in tqdm(range(len(rotates))):
        #     # frame = np.concatenate([
        #     #         np.concatenate([rotates[t], outputs[t,:,:]], axis=0),
        #     #         np.concatenate([outputs[:,t,:], outputs[:,:,t]], axis=0)
        #     # ], axis=1)

        #     # frame = (255 * frame).astype(np.uint8)
        #     frame = image
        #     frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        #     # frame = image
        #     video_file.write(frame)
        # video_file.release()

        return video_name
       
    # Load the Lena image
    # image = cv2.imread("data/xr/0a8dc3de200bc767169b73a6ab91e948.png")

    # Create Gradio interface
    iface = gr.Interface(
        fn=image_to_volume,
        inputs=gr.Image(type="numpy", image_mode="L"),
        examples=[
            # "data/ChestXRLungSegmentation/VinDr/v1/processed/test/images/0a8dc3de200bc767169b73a6ab91e948.png",
            # "data/ChestXRLungSegmentation/VinDr/v1/processed/test/images/0a2e45455fe2c76ddbbb5f38ab35f6f2.png",
            # "data/ChestXRLungSegmentation/VinDr/v1/processed/test/images/0a6fd1c1d71ff6f9e0f0afa746e223e4.png",
            "data/ChestXRLungSegmentation/VinDr/v1/processed/test/images/0b2cc81ad04ca2e91f2a8626b645cad8.png",
            "data/ChestXRLungSegmentation/VinDr/v1/processed/test/images/1b4407d5e1e9ec1e602c75fc80b7b001.png",
        ],
        outputs=gr.Video()
    )

    # Launch the app
    iface.launch()

if __name__ == "__main__":
    main()