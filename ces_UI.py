import gradio as gr
import yaml
import torch

from diff_renderer.diff_optimizer_recon import Optimizer_recon
from diff_renderer.optimizer_mocap import Optimizer_mocap
from fbx_utils.fbx_process import FBX_Generator

theme = gr.themes.Ocean(
    primary_hue="fuchsia",
    secondary_hue="blue",
    neutral_hue="indigo",)

# GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reset_fields():
    return None, None, None

def process_mocap(input_path, pose_path):
    # load params
    path2config_mocap = './config/exp_config_mocap.yaml'
    with open(path2config_mocap, 'r') as f:
        params_mocap = yaml.load(f, Loader=yaml.FullLoader)

    optimizer = Optimizer_mocap(params_mocap, device=device)
    input_dict = dict()

    file_name = input_path.split('/')[-1]
    data_name = file_name[0:-10]
    input_dict["input_path"] = input_path
    input_dict["input_pose"] = pose_path
    input_dict["data_name"] = data_name
    input_dict["file_name"] = file_name
    input_dict["save_path"] = './results'
    input_dict["skip_exist"] = False

    if input_dict is None:
        return "No input dictionary uploaded or file is invalid. Please upload a valid input dictionary."
    try:
        out_path = optimizer.forward(input_dict)
        return out_path

    except Exception as e:
        return f"An error occurred: {str(e)}"

def process_recon(input_path, smpl_path):
    # load params
    path2config_recon = './config/exp_config_recon.yaml'
    with open(path2config_recon, 'r') as f:
        params_recon = yaml.load(f, Loader=yaml.FullLoader)

    optimizer = Optimizer_recon(params_recon, device=device)
    input_dict = dict()

    file_name = input_path.split('/')[-1]
    data_name = file_name[0:-10]
    params_recon['DATA']['data_name'] = data_name
    params_recon['DATA']['file_name'] = file_name
    input_dict["input_path"] = input_path
    input_dict["input_smplx"] = smpl_path
    input_dict["save_path"] = './results'

    if input_dict is None:
        return "No input dictionary uploaded or file is invalid. Please upload a valid input dictionary."
    
    try:
        optimizer.pred_uv_disp(input_dict)
        optimizer.smplx_base_recon()
        out_path = optimizer.pipeline()
        
        return out_path

    except Exception as e:
        return f"An error occurred: {str(e)}"

def process_animation(input_path, motion_path):
    # # load params
    # path2config_mocap = './config/exp_config_mocap.yaml'
    # with open(path2config_mocap, 'r') as f:
    #     params_mocap = yaml.load(f, Loader=yaml.FullLoader)

    fbx_gen = FBX_Generator(device=device)
    input_dict = dict()

    # # load
    # with open('data.pickle', 'rb') as f:
    #     data = pickle.load(f)

    file_name = input_path.split('/')[-1]
    data_name = file_name[0:-14]
    input_dict["input_path"] = input_path
    input_dict["input_motion"] = motion_path
    input_dict["data_name"] = data_name
    input_dict["file_name"] = file_name
    input_dict["save_path"] = './results'

    if input_dict is None:
        return "No input dictionary uploaded or file is invalid. Please upload a valid input dictionary."
    try:
        out_path = fbx_gen.forward(input_dict)
        return out_path

    except Exception as e:
        return f"An error occurred: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # print logos
    with gr.Row():
        with gr.Column():
            gr.Image(value="logos/integ_logo.png",
                     show_download_button=False,
                     container=False,
                     show_fullscreen_button=False,
                     show_label=False, elem_id="logo-img", width=200, height=200)
        # gr.Image(value="logos/polygom_thumbnail.jpg",
        #          show_download_button=False,
        #          container=False,
        #          show_fullscreen_button=False,
        #          show_label=False, elem_id="logo-img", width=1000, height=200)
    with gr.Row():
        gr.Markdown("### We are creating an intuitive solution that allows non-experts to easily generate 3D human models using just images or text inputs.")

    # pre-processing & mocap tab
    with gr.Tab("Step1: Motion Capture"):
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="Input Image", type='filepath')
                input_pose = gr.File(label="Input SMPL params", type='filepath')
                # input_img = gr.Image(sources=["webcam"], type="numpy")

                with gr.Row():
                    with gr.Column(scale=1.0, min_width=50):
                        process_button = gr.Button(value="Run")
                    with gr.Column(scale=1.0, min_width=50):
                        reset_button = gr.Button(value="Clear")
                with gr.Row():
                    examples_image = gr.Examples(examples=[["examples/jh_input.png", "examples/jh_pose.json"]],
                                                 inputs=[input_img, input_pose])
            with gr.Column(scale=4.0):
                result = gr.Model3D(label="3d mesh reconstruction")
        process_button.click(fn=process_mocap, inputs=[input_img, input_pose], outputs=[result])
        reset_button.click(fn=reset_fields, inputs=[], outputs=[input_img, input_pose, result])
            # with gr.Column():
            #     output_img = gr.Image(streaming=True)
            # dep = input_img.stream(process_input, [input_img, transform], [output_img],
            #                        time_limit=30, stream_every=0.1, concurrency_limit=30)

    # reconstruction & optimizing tab
    with gr.Tab("Step2: Image to 3D Mesh"):
        with gr.Row():
                with gr.Column(scale=1.0):
                    image_file = gr.Image(label="Input Image", type='filepath')
                    smpl_file = gr.File(label="Input SMPL params", type='filepath')
                    with gr.Row():
                        with gr.Column(scale=1.0, min_width=50):
                            process_button = gr.Button(value="Run")
                        with gr.Column(scale=1.0, min_width=50):
                            reset_button = gr.Button(value="Clear")
                    with gr.Row():
                        examples_image = gr.Examples(examples=[["examples/mg_input.png", "examples/standard_mg.json"],
                                                               ["examples/mh_input.png", "examples/standard_mh.json"]],
                                                     inputs=[image_file, smpl_file])
                with gr.Column(scale=4.0):
                    result = gr.Model3D(label="3d mesh reconstruction")

        process_button.click(fn=process_recon, inputs=[image_file, smpl_file], outputs=[result])
        reset_button.click(fn=reset_fields, inputs=[], outputs=[image_file, smpl_file, result])

    # animation tab
    with gr.Tab("Step3: Animating Reconstructed 3D Mesh"):
        with gr.Row():
            with gr.Column():
                input = gr.Image(label="Input Image", type='filepath')
                # input = gr.Model3D(label="3d mesh reconstruction")
                motion_file = gr.File(label="Input motion", type='filepath')

                with gr.Row():
                    with gr.Column(scale=1.0, min_width=50):
                        process_button = gr.Button(value="Run")
                    with gr.Column(scale=1.0, min_width=50):
                        reset_button = gr.Button(value="Clear")
                with gr.Row():
                    examples_image = gr.Examples(examples=[["examples/jh_input.png", "examples/t2m-gpt-motion.pkl"]],
                                                 inputs=[input, motion_file])
            with gr.Column(scale=4.0):
                result = gr.Model3D(label="3d mesh reconstruction")
        process_button.click(fn=process_animation, inputs=[input, motion_file], outputs=[result])
        reset_button.click(fn=reset_fields, inputs=[], outputs=[input, motion_file, result])
        # with gr.Row():
        #     gr.HTML("<img src='path/to/img.png'")

    with gr.Row():
        gr.Image(value="logos/partners.png",
                 show_download_button=False,
                 container=False,
                 show_fullscreen_button=False,
                 show_label=False, elem_id="logo-img", width=100, height=80)

demo.launch(auth=("polygom", "0000"))