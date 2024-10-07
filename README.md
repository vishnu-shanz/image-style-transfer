Image Style Transfer

Description

This project implements Image Style Transfer using deep learning. The goal is to blend two images together: one image will provide the content (e.g., a photo), and the other will provide the style (e.g., a painting). This technique allows us to create artistic images by combining the content of one image with the artistic style of another.

The implementation uses a pre-trained convolutional neural network (VGG19) and the process is based on the paper "A Neural Algorithm of Artistic Style" by Gatys et al.

Features

Combine any content image with a style image.

Fine-tune the ratio of content to style representation.

Supports multiple optimization techniques.

Easily modify hyperparameters such as learning rate and iterations.

Output visually appealing and high-quality images.


How It Works

1. Load Pre-trained Model: The project uses a pre-trained VGG19 network.


2. Extract Features: Features from different layers of the network are extracted to capture content and style representations.


3. Optimization: The content image is updated to minimize the difference between its features and those of the target style and content.



Requirements

Python 3.7+

PyTorch

Matplotlib

Pillow (for image processing)

Numpy

Torchvision
Install Dependencies

pip install -r requirements.txt

Usage

1. Clone the repository:



git clone https://github.com/vishnu_shanz/Image-Style-Transfer.git
cd Image-Style-Transfer

2. Run the Style Transfer Script:



python style_transfer.py --content <path_to_content_image> --style <path_to_style_image> --output <output_image_path>

Example:

python style_transfer.py --content images/photo.jpg --style images/painting.jpg --output results/stylized.jpg

3. Adjust Parameters: You can fine-tune the output by adjusting parameters like the content and style weight:



python style_transfer.py --content images/photo.jpg --style images/painting.jpg --output results/stylized.jpg --content_weight 1e5 --style_weight 1e10

Sample Results

Acknowledgements

This project is inspired by the paper "A Neural Algorithm of Artistic Style" by Gatys et al.

Pre-trained models provided by the PyTorch model zoo.


Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

License

This project is licensed under the MIT License - see the LICENSE file for details.
