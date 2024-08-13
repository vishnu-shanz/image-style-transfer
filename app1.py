import torch
import process_stylization
from photo_wct import PhotoWCT
import gradio as gr

# Load model
model_path = './models/photo_wct.pth'
p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load(model_path))

def run(content_img, style_img, post_processing, algorithm):
    p_pro = None

    if algorithm == "Guided Image Filtering (Fast)":
        from photo_smooth import Propagator
        p_pro = Propagator()
    elif algorithm == "Photorealistic Smoothing (Slow)":
        from photo_gif import GIFSmoothing
        p_pro = GIFSmoothing(r=35, eps=0.001)
    elif algorithm == "VGG16":
        from style_transfer import style_transfer
        return style_transfer(content_img, style_img, post_processing)
    p_wct.to('cpu') 
    
    output_img = process_stylization.stylization_gradio(
        stylization_module=p_wct,
        smoothing_module=p_pro,  
        content_image=content_img,
        style_image=style_img,
        post_processing=post_processing
    )
    
    return output_img

if __name__ == '__main__':
    style = gr.Interface(
        fn=run, 
        inputs=[
            gr.Image(show_label='Content Image',label='content_img'),
            gr.Image(show_label='Stylize Image',label='style_img'),
            gr.Checkbox(value=True, show_label='Post Processing',label='post_processing'),
            gr.Radio(["Guided Image Filtering (Fast)", "Photorealistic Smoothing (Slow)", "VGG16"],show_label="Choose Algorithm",label='algorithm')
        ], 
        outputs=[
            gr.Image(type="pil", label="Result"),
        ]    
    )
    style.queue()
    style.launch()
