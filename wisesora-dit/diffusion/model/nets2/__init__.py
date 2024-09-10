from .latte_t2v import LatteT2V
from .latte_t2v_debug import LatteT2V_Debug
from .pipeline_videogen import VideoGenPipeline

def get_models(args):
    pretrained_model_path = "./configs"
    return LatteT2V.from_pretrained_2d(pretrained_model_path, subfolder="transformer", video_length=args.video_length)

def get_debug_models(args):
    pretrained_model_path = "./configs"
    return LatteT2V_Debug.from_pretrained_2d(pretrained_model_path, subfolder="transformer", video_length=args.video_length)