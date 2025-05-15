from PatchEmbedding import PatchEmbedding
import torch.nn as nn
import torch

from visiontransformer.TransformerEncoder import TransformerEncoder


class VisionTransformer(nn.Module):
    def __init__(
      self,
      img_size = 224,
      patch_size = 16,
      in_channels = 3,
      num_classes = 1000,
      embed_dim = 768,
      num_heads = 12,
      num_layers = 12,
      ffn_hidden = None,
      drop_out = 0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size,patch_size,in_channels,embed_dim)
        self.num_patches = self.patch_embed.num_patches

        #CLS token
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))

        #Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1,self.num_patches+1,embed_dim))
        self.pos_drop = nn.Dropout(p=drop_out)

        #Transformer encoder
        self.encoder = TransformerEncoder(embed_dim,num_heads,ffn_hidden,num_layers,drop_out)

        #MLP head
        self.mlp_head = nn.Linear(embed_dim,num_classes)

    def forward(self,x):
        x = self.patch_embed(x)
        #patchEmbedding
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B,-1,-1) #[B,1,D]
        x = torch.cat([cls_tokens,x],dim=1) #[B,N+1,D]
        x =x+ self.pos_embed
        x = self.pos_drop(x)

        x = self.encoder(x)

        cls_output = x[:,0] #get CLS Token [B,D]
        logits = self.mlp_head(cls_output)  #classification [B,num_classes]

        return logits

model = VisionTransformer(
    img_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=10,    # 举个例子：10类分类任务
    embed_dim=768,
    num_heads=12,
    num_layers=12,
    ffn_hidden=3072,
    drop_out=0.1
)

dummy_input = torch.randn(2, 3, 224, 224)  # [B, C, H, W]
output = model(dummy_input)  # 应该输出 [2, 10]
print(output.shape)