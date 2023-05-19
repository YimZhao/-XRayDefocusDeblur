import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import denoising_diffusion_pytorch as ddp
import dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define the denoising diffusion model
model = ddp.Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).to(device)

diffusion = ddp.GaussianDiffusion(
    model,
    image_size = 512,
    timesteps = 10,   # number of steps
    loss_type = 'l2'    # L1 or L2
).to(device)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Define the loss function
loss_fn = nn.MSELoss()

# Define the dataset and dataloader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainpath = './Cell_gen/v4/rnd_pyramid/gt'
train_dataset = dataset.MyCustomDatasetTest(trainpath,transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

# Train the model
for epoch in range(10):
    for i, images in enumerate(train_loader):
        # Denoise the images using the denoising diffusion model
        images = images.to(device)
        denoised_images = diffusion(images)

        # Compute the loss
        loss = loss_fn(denoised_images, images)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss every 100 batches
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{10}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Save the model checkpoint every epoch
        if (i+1) == len(train_loader):
            checkpoint_path = f"./diffusion/checkpoint_epoch_{epoch+1}.ckpt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at epoch {epoch+1}")
