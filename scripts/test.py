import torch
import sys
sys.path.append('/home/lijunjie/work/PixArt-alpha')
from diffusion.model.nets.PixArt import PixArt_XL_2
from diffusion.model.nets.PixArt_t2v import PixArt_XL_2_T2V
from diffusion.model.t5 import T5Embedder
from diffusion.model.utils import prepare_prompt_ar
import numpy as np
from diffusion.data.datasets import ASPECT_RATIO_512_TEST


if __name__ == "__main__":
    # 测试原模型
    # 1、初始化1张图
    # x = torch.randn([1,3,512,512]).cuda()
    # image_size = 512
    # latent_size = 512 // 8
    # lewei_scale = 1
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = PixArt_XL_2(input_size=latent_size, lewei_scale=lewei_scale).to(device)
    # model_path = "../output/pretrained_models/PixArt-XL-2-512x512.pth"
    # state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    # del state_dict['state_dict']['pos_embed']
    # missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
    # model.eval()
    # latent_size_h, latent_size_w = latent_size, latent_size
    # n = 1
    # latents = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
    # # ts = torch.from_numpy(np.arange(0, 1000)[::-1].copy().astype(np.int64)).to(device)
    # bs = 1
    # t = 1
    # ts = torch.full((bs,), t, device=device, dtype=torch.long)
    # hw = torch.tensor([[512, 512]], dtype=torch.float, device=device).repeat(bs, 1)
    # ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
    # caption_embs, emb_masks = torch.randn([1,1,120,4096]).to(device), torch.randn([1,1,120]).to(device)
    # caption_embs = caption_embs.float()[:, None]
    # model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
    # model_pred = model(latents, ts, caption_embs, **model_kwargs)[:, :4]
    # print(model_pred.shape)

    # 1、初始化latent [1,6,4,64,64]
    image_size = 512
    latent_size = 512 // 8
    lewei_scale = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PixArt_XL_2_T2V(input_size=latent_size, lewei_scale=lewei_scale).to(device)
    model_path = "output/pretrained_models/PixArt-XL-2-512x512.pth"
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    del state_dict['state_dict']['pos_embed']
    missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
    model.eval()
    latent_size_h, latent_size_w = latent_size, latent_size
    n = 1
    bs = 1
    t = 1
    latents = torch.randn(n, 6, 4, latent_size_h, latent_size_w, device=device)
    ts = torch.full((bs,), t, device=device, dtype=torch.long)
    hw = torch.tensor([[512, 512]], dtype=torch.float, device=device).repeat(bs, 1)
    ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
    caption_embs, emb_masks = torch.randn([1,35,4096]).to(device), None
    caption_embs = caption_embs.float()[:, None]
    model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
    t5_path = "output/pretrained_models/t5_ckpts"
    base_ratios = eval('ASPECT_RATIO_512_TEST')
    t5 = T5Embedder(device="cuda", local_cache=True, cache_dir=t5_path, torch_dtype=torch.float)
    # batchsize*use_image_num
    y_image = [['A small cactus with a happy face in the Sahara desert.','professional portrait photo of an anthropomorphic cat wearing fancy gentleman hat and jacket walking in autumn forest.', 'an astronaut sitting in a diner, eating fries, cinematic, analog film','Pirate ship trapped in a cosmic maelstrom nebula, rendered in cosmic beach whirlpool engine, volumetric lighting, spectacular, ambient lights, light pollution, cinematic atmosphere, art nouveau style, illustration art artwork by SenseiJaye, intricate detail.']]
    all_captions_embs = []
    for batch in range(bs):
        batch_captions_embs = []
        for caption in y_image[batch]:
            prompts = []
            prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(caption, base_ratios, device=device, show=False)
            prompts.append(prompt_clean.strip())
            caption_embs, emb_masks = t5.get_text_embeddings(prompts)
            caption_embs = caption_embs.float()[:, None]
            batch_captions_embs.append(caption_embs)
        all_captions_embs.append(batch_captions_embs)
    model_pred = model(latents, ts, caption_embs, use_image_num=4, y_image = all_captions_embs, **model_kwargs)[:, :4]
    print(model_pred.shape)