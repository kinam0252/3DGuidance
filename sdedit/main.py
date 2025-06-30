import cv2
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video, load_video

# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = "/home/nas_main/kinamkim/.checkpoint/Wan2.1-T2V-14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

video_path = "src.mp4"
video = load_video(video_path)

image = video[0] # pil image
width, height = image.size

@torch.no_grad()
def get_video_latents(pipe, video, generator, dtype, height, width):
    
    from diffusers.utils import load_video
    from diffusers.pipelines.wan.pipeline_wan_video2video import retrieve_latents
    from diffusers.utils.torch_utils import randn_tensor
    video = pipe.video_processor.preprocess_video(video, height=height, width=width)
    video = video.to("cuda", dtype=pipe.vae.dtype)
    
    video_latents = retrieve_latents(pipe.vae.encode(video))
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(pipe.device, dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
        pipe.device, dtype
    )

    init_latents = (video_latents - latents_mean) * latents_std

    init_latents = init_latents.to(pipe.device)
    return init_latents

generator = torch.manual_seed(42)
init_latents = get_video_latents(pipe, video, pipe.dtype, height, width)

generator = torch.manual_seed(42)

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=49,
    guidance_scale=5.0,
    init_latents=init_latents,
).frames[0]
export_to_video(output, "output.mp4", fps=8)
