import torch
import onnx

saved_model_path = 'data/output_tensorboard/final_model.pth'
model = torch.load(saved_model_path, weights_only=False)
model.eval()

dummy_input = torch.rand(1, 3, 256, 256, device='mps')
torch.onnx.export(
    model,
    dummy_input,
    'data/convert_pth.onnx',
    # verbose=True,
    input_names=['conv1'],
    output_names=['fc']
)

model = onnx.load('data/convert_pth.onnx')
onnx.checker.check_model(model)
# print(onnx.helper.printable_graph(model.graph))