from pytorch3d.io import IO
import torch
device=torch.device("cuda:0")
mesh = IO().load_mesh("final_model.obj", device=device)
