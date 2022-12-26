import torch
import gradio as gr
from src.predict import predict_interval, load_torch_net


def predict_gradio_canvas(x, net, device="cuda"):
    if x is None:
        return {0: 0}
    else:
        x = torch.from_numpy(x.reshape(1, 28, 28)).to(dtype=torch.float32, device=device)
        return predict_interval(x, net, device)


def main(device="cuda"):
    net = load_torch_net("../checkpoints/pytorch/version_1.pt")

    gr.Interface(fn=lambda x: predict_gradio_canvas(x, net, device),
                 inputs="sketchpad",
                 outputs="label",
                 live=True).launch()


if __name__ == "__main__":
    main(device="cpu")
