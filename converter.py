import io
import numpy as np
import torch.onnx
from model import U2NET

torch_model = U2NET(3,1)
model_path = "u2net_bce_itr_2000_train_2.445896_tar_0.330302.pth"
batch_size = 1

torch_model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
torch_model.eval()

x = torch.randn(batch_size, 3, 320, 320, requires_grad=True)
torch_out = torch_model(x)

torch.onnx.export(torch_model, x, "new-model.onnx", export_params=True, opset_version=11, do_constant_folding=True, input_names = ['input'], output_names = ['output'], dynamic_axes = {'input' : {0: 'batch_size'}, 'output': {0: 'batch_size'}})