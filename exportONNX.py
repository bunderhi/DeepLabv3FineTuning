import torch

model = torch.load('/models/run03/weights.pt')
model = model.cuda().eval().half()
x = torch.ones(1, 3, 230, 500, requires_grad=True).cuda().half()
torch_out = model(x)
torch.onnx.export(model,x,'/models/run03/jetracer.onnx',export_params=True,opset_version=11,do_constant_folding=True,
                    input_names=['input'],output_names=['output'])
