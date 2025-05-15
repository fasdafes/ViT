from torch import nn

class PatchEmbedding(nn.Module):
  def __init__(self,img_size,patch_size = 16,in_channels=3,embed_dim = 768):
    super().__init__()
    self.patch_size = patch_size
    self.num_patches = (img_size//patch_size)**2
    self.proj = nn.Conv2d(in_channels,embed_dim,kernel_size=patch_size,stride=patch_size)
    """
    this conv2d layer is used to flatten the image into patches
    a img with size of (224,224,3) will be flattened into (14,14,768)
    why?
    Cuz,it's conved by a kernerl witch size of 16*16 and stride of 16,so 224 divided by 16 is 14
    and 768 kernels are used to flatten the image
    """
  def forward(self,x):
    x = self.proj(x)
    x = x.flatten(2)
    x = x.transpose(1,2)
    return x