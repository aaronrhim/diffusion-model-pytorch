import cv2
import time
import torch
import numpy as np
from collections import deque
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import gc

# Build model and diffusion process.
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=True
)

diffusion = GaussianDiffusion(
    model,
    image_size=128,
    timesteps=1000,           # total diffusion steps
    sampling_timesteps=250    # accelerated (DDIM) sampling steps
)

# (Optional) Load pretrained weights if available.
# model.load_state_dict(torch.load('path/to/your/trained_model.pth'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
diffusion.sqrt_alphas_cumprod = diffusion.sqrt_alphas_cumprod.to(device)
print(diffusion.sqrt_alphas_cumprod.device)
print(device)

# Define an optimizer for online training.
optimizer = torch.optim.Adam(model.parameters(), lr=8e-5)

# Use a smaller buffer (e.g., 50 frames instead of 150) to reduce batch size.
buffer = deque(maxlen=50)

# Initialize the webcam.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

train_interval = 10  # seconds
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess: Convert from BGR to RGB and resize to 128x128.
    #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame_resized = cv2.resize(frame_rgb, (128, 128))
    frame_resized = cv2.resize(frame, (128, 128))
    frame_tensor = torch.tensor(frame_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
    frame_tensor.to(device)
    buffer.append(frame_tensor)

    if time.time() - start_time >= train_interval and len(buffer) > 0:
        batch = torch.stack(list(buffer)).to(device)
        #t = torch.randint(0, diffusion.num_timesteps, (batch.size(0),), device=device).long()
        diffusion.sqrt_alphas_cumprod = diffusion.sqrt_alphas_cumprod.to(device)
        diffusion.sqrt_one_minus_alphas_cumprod = diffusion.sqrt_one_minus_alphas_cumprod.to(device)
        t = torch.randint(0, diffusion.num_timesteps, (batch.size(0),), device=device).long()
        noisy_batch = diffusion.q_sample(batch, t)
        print("test model")
        pred_noise = model(noisy_batch, t)
        loss = ((pred_noise - (noisy_batch - batch)) ** 2).mean()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Trained on buffered batch, loss: {loss.item():.4f}")
        
        # Reset the buffer and timer.
        buffer.clear()
        start_time = time.time()
        
        # Pause briefly to let the system catch up.
        time.sleep(5)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    # Optionally, display the live feed.
    cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
