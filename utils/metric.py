import torch
import fastmri

# --------------------------------
# metric
# --------------------------------
def abs(x):
    return fastmri.complex_abs(x.squeeze().permute(1,2,0)).detach().cpu().numpy()

def cal_psnr(a, b, max_pixel=1, complex=False):
    with torch.no_grad():
        if complex:
            a = fastmri.complex_abs(a.permute(0, 2, 3, 1))
            b = fastmri.complex_abs(b.permute(0, 2, 3, 1))
        mse = torch.mean((a - b) ** 2)
        if mse == 0:
            psnr = 100
        else:
            psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.detach().cpu().numpy() if psnr.device is not 'cpu' else psnr