"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import torch

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if torch.cuda.is_available():
        model = torch.load("{}/flappy_bird".format(opt.saved_path))
    else:
        model = torch.load("{}/flappy_bird".format(opt.saved_path), map_location=lambda storage, loc: storage)
    
    model.eval()
    """
    In PyTorch, the `model.eval()` method is used to set the model to evaluation mode. This mode is typically used during inference or evaluation, rather than during training. When the model is in evaluation mode, several things happen:

1. **Batch Normalization and Dropout Layers:** Batch normalization and dropout layers, which behave differently during training and evaluation, are set to evaluation mode. During training, batch normalization calculates the batch statistics (mean and variance) for each batch and applies dropout to the input. In evaluation mode, batch normalization uses the running statistics (accumulated statistics from training) and does not apply dropout, ensuring deterministic behavior.

2. **Autograd Tracking:** Setting the model to evaluation mode disables autograd tracking. This means that operations on tensors will not be recorded for automatic differentiation, which is used for computing gradients during training. Disabling autograd tracking reduces memory consumption and speeds up inference, as gradients are not needed during inference.

3. **Other Layers:** Some layers may have different behaviors during training and evaluation. For example, some layers may apply different scaling factors or activation functions. Setting the model to evaluation mode ensures that these layers behave consistently during inference.

Overall, `model.eval()` is used to ensure that the model behaves consistently during inference, with specific considerations for batch normalization, dropout, and autograd tracking.
    """
    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image)
    
    
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    while True:
        prediction = model(state)[0]
        action = torch.argmax(prediction).item()

        next_image, reward, terminal = game_state.next_frame(action)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size,
                                    opt.image_size)
        next_image = torch.from_numpy(next_image)
        
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

        state = next_state


if __name__ == "__main__":
    opt = get_args()
    test(opt)
