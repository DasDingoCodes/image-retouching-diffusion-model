import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, channels, size, dropout=0.0):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, dropout=0.0):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout(p=dropout),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout(p=dropout),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, dropout=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True, dropout=dropout),
            DoubleConv(in_channels, out_channels, dropout=dropout),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, dropout=0.0):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True, dropout=dropout),
            DoubleConv(in_channels, out_channels, in_channels // 2, dropout=dropout),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class ConditionalInjection(nn.Module):
    def __init__(self, x_in_channels, conditional_in_channels, out_channels, downscaling_factor, dropout=0.0, pooling_operation="avg", residual=True):
        super().__init__()
        self.residual = residual
        if pooling_operation == "avg":
            self.downscaling = nn.AvgPool2d(downscaling_factor)
        elif pooling_operation == "max":
            self.downscaling = nn.MaxPool2d(downscaling_factor)
        else:
            self.downscaling = nn.AvgPool2d(downscaling_factor)
        self.conv = DoubleConv(x_in_channels + conditional_in_channels, out_channels, dropout=dropout)
    
    def forward(self, x, conditional):
        conditional = self.downscaling(conditional)
        x_conditional = torch.concat((x, conditional), dim=1)
        if self.residual:
            return F.gelu(x + self.conv(x_conditional))
        else:
            return self.conv(x_conditional)

class VectorToSquareMatrix(nn.Module):
    def __init__(self, embedding_dim, target_dim, max_start_dim=16, hidden=64):
        super().__init__()
        scaling_steps = self._num_scaling_steps(max_start_dim, target_dim)
        self.start_dim = target_dim // 2**scaling_steps
        self.embedding_in = nn.Linear(embedding_dim, self.start_dim*self.start_dim)
        upscaling_list = [
            nn.ConvTranspose2d(
                1,
                hidden,
                2,
                stride=2,
            )
        ]
        for i in range(scaling_steps - 1):
            upscaling_list.append(
                nn.ConvTranspose2d(
                    hidden,
                    hidden,
                    2,
                    stride=2,
                )
            )
        self.upscaling = nn.Sequential(*upscaling_list)
    
    def _num_scaling_steps(self, dim_small, dim_big):
        pow_2_dif = math.log2(dim_big) - math.log2(dim_small)
        return math.ceil(pow_2_dif)

    def forward(self, x: torch.Tensor):
        x = self.embedding_in(x)
        x = x.view(-1, self.start_dim, self.start_dim)
        x = x.unsqueeze(1)
        x = self.upscaling(x)
        return x

class UNetExtended(nn.Module):
    def __init__(
        self, 
        img_shape=(256, 256),
        input_channels=3,
        output_channels=3,
        time_dim=256, 
        hidden=64, 
        level_mult = [1,2,4,8],
        num_middle_layers = 3,
        num_patches=4,
        use_self_attention=True,
        use_conditional_image=False,
        conditional_img_channels=3,
        use_conditional_embedding=False,
        embedding_dim=384,
        device="cuda",
        dropout=0.0,
        output_activation_func="tanh",
        activation_func="relu",
    ):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.level_mult = level_mult
        self.num_patches = num_patches
        self.use_self_attention = use_self_attention
        self.use_conditional_image = use_conditional_image
        self.conditional_img_channels = conditional_img_channels * num_patches * num_patches # conditional image gets divided into patches
        self.use_conditional_embedding = use_conditional_embedding
        self.embedding_dim=embedding_dim
        self.dropout = dropout

        self.num_levels = len(self.level_mult) - 1


        self.input_channels = input_channels
        self.output_channels = output_channels        
        self.height, self.width = img_shape
        assert self.height % num_patches == 0, "height of input images needs to be divisible by number of patches"
        assert self.width % num_patches == 0, "width of input images needs to be divisible by number of patches"
        assert self.height == self.width, "only allow square images for now"
        self.img_size = self.height

        # window size when combining patches (after conv layers)
        self.cp_window_size = (num_patches, num_patches)
        self.cp_stride = self.cp_window_size

        # after patch extraction and reshaping the dimension of the first Convs input channel will be:
        # channels * num_patches * num_patches
        self.c_in = self.input_channels * num_patches * num_patches
        self.c_out = self.output_channels * num_patches * num_patches

        self.in_layer = DoubleConv(self.c_in, hidden * level_mult[0], dropout=dropout)
        if self.use_conditional_image:
            self.in_layer_conditional_img = ConditionalInjection(
                hidden * level_mult[0],
                self.conditional_img_channels,
                hidden * level_mult[0],
                1, # don't need to downscale conditional image yet
                dropout=dropout
            )
        
        if self.use_conditional_embedding:
            self.embedding_reshaping = VectorToSquareMatrix(
                self.embedding_dim,
                self.img_size, # make vector embedding matrix as big as input/output image
                hidden=hidden
            )
            self.in_layer_conditional_vector = ConditionalInjection(
                hidden * level_mult[0],
                hidden, # the vector embeddings get reshaped to a matrix with channel size of hidden
                hidden * level_mult[0],
                self.num_patches, 
                dropout=dropout
            )

        level = 0
        self.down_conv_layers = []
        self.down_att_layers = []
        self.down_conditional_img_layers = []
        self.down_conditional_txt_layers = []
        for i in range(self.num_levels):
            level += 1
            hidden_in = hidden * level_mult[i]
            hidden_out= hidden * level_mult[i+1]
            self.down_conv_layers.append(
                Down(hidden_in, hidden_out, dropout=dropout)
            )
            # conditional image layers take in x and the conditional image
            # they have to know the output channels of the previous layer (hidden_out),
            # the channels of the conditional image (conditional_img_channels)
            # the conditional layer shold output the same shape as the previous layer
            # so it too will output hidden_out
            # the conditional image has to be scaled down to the size of the x at that layer
            # the downscaling factor will be num_patches * 2^x with x being the "level" the layer is at within the UNet
            if self.use_conditional_image:
                self.down_conditional_img_layers.append(
                    ConditionalInjection(
                        hidden_out,
                        self.conditional_img_channels,
                        hidden_out,
                        2**level, 
                        dropout=dropout
                    )
                )
            if self.use_conditional_embedding:
                self.down_conditional_txt_layers.append(
                    ConditionalInjection(
                        hidden_out,
                        hidden,
                        hidden_out,
                        self.num_patches * 2**level, 
                        dropout=dropout,
                        pooling_operation="max"
                    )
                )
            # self attention outputs shape (B, C, S, S) with C channels and S size
            # channels is the same as the layer before self attention outputs
            # i.e. (if we ignore the conditional layer) what is given as the second parameter of Down instance (hidden_out)
            # size is also the same as the previous layer outputs but we first have to calculate the size
            # it is the input image size / num_patches / 2^x with x being the "level" the layer is at within the UNet
            if self.use_self_attention:
                self.down_att_layers.append(
                    SelfAttention(hidden_out, self.img_size // self.num_patches // 2**level, dropout=dropout)
                )
        self.down_conv_layers = nn.ModuleList(self.down_conv_layers)
        self.down_conditional_img_layers = nn.ModuleList(self.down_conditional_img_layers)
        self.down_conditional_txt_layers = nn.ModuleList(self.down_conditional_txt_layers)
        self.down_att_layers = nn.ModuleList(self.down_att_layers)
        
        self.middle_layers = []
        hidden_middle = hidden * level_mult[-1]
        # for _ in range(num_middle_layers):
        #     self.middle_layers.append(DoubleConv(hidden_middle, hidden_middle))
        for _ in range(num_middle_layers):
            self.middle_layers.append(DoubleConv(hidden_middle, hidden_middle, dropout=dropout))
        # # channel count of last middle layer has to fit the channel count of the skip connection with the lowest level
        # # i.e. hidden * factor of second last level
        # self.middle_layers.append(DoubleConv(hidden_middle, hidden * level_mult[-2]))
        self.middle_layers = nn.ModuleList(self.middle_layers)
        
        reversed_level_mult = list(reversed(level_mult))
        self.up_conv_layers = []
        self.up_att_layers = []
        self.up_conditional_img_layers = []
        self.up_conditional_txt_layers = []
        for i in range(self.num_levels):
            level -= 1
            # hidden in takes in the output of the previous layer 
            # (for the first UP it's the last middle layer, for the other UP layers its the respective previous UP)
            # and hidden in takes in some skip connection
            # (UP on level n takes in the same x as the DOWN on level n)
            hidden_in = hidden * reversed_level_mult[i] + hidden * reversed_level_mult[i+1]
            hidden_out= hidden * reversed_level_mult[i+1]
            self.up_conv_layers.append(
                Up(hidden_in, hidden_out, dropout=dropout)
            )
            if self.use_conditional_image:
                self.up_conditional_img_layers.append(
                    ConditionalInjection(
                        hidden_out,
                        self.conditional_img_channels,
                        hidden_out,
                        2**level, 
                        dropout=dropout
                    )
                )
            if self.use_conditional_embedding:
                self.up_conditional_txt_layers.append(
                    ConditionalInjection(
                        hidden_out,
                        hidden,
                        hidden_out,
                        self.num_patches * 2**level, 
                        dropout=dropout,
                        pooling_operation="max"
                    )
                )
            if self.use_self_attention:
                self.up_att_layers.append(
                    SelfAttention(hidden_out, self.img_size // self.num_patches // 2**level, dropout=dropout)
                )
        self.up_conv_layers = nn.ModuleList(self.up_conv_layers)
        self.up_conditional_img_layers = nn.ModuleList(self.up_conditional_img_layers)
        self.up_conditional_txt_layers = nn.ModuleList(self.up_conditional_txt_layers)
        self.up_att_layers = nn.ModuleList(self.up_att_layers)
        if self.use_conditional_image:
            self.out_layer_conditional_img = ConditionalInjection(
                hidden * level_mult[0],
                self.conditional_img_channels,
                hidden * level_mult[0],
                1, # don't need to scale down conditional image
                dropout=dropout
            )
        if self.use_conditional_embedding:
            self.out_layer_conditional_vector = ConditionalInjection(
                hidden * level_mult[0],
                hidden,
                hidden * level_mult[0],
                self.num_patches, 
                dropout=dropout
            )

        self.out_layer = nn.Conv2d(hidden * level_mult[0], self.c_out, kernel_size=3, padding=1, bias=False)
        self.output_activation_func = self.activation_func_from_str(output_activation_func)
        self.activation_func = self.activation_func_from_str(activation_func)

    def activation_func_from_str(self, name: str):
        if name == "tanh":
            return nn.Tanh()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "relu":
            return nn.ReLU()
        else:
            return None

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def to_patches(self, x, patch_size=2):
        """Splits tensor x into patches_size*patches_size patches"""
        p = patch_size
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H, W//p, C*p)
        x = x.permute(0, 2, 1, 3).reshape(B, W//p, H//p, C*p*p)
        return x.permute(0, 3, 2, 1)

    def from_patches(self, x, patch_size=2):
        """Combines x's patches_size*patches_size patches into one"""
        p = patch_size
        B, C, H, W = x.shape
        x = x.permute(0,3,2,1).reshape(B, W, H*p, C//p)
        x = x.permute(0,2,1,3).reshape(B, H*p, W*p, C//(p*p))
        return x.permute(0, 3, 1, 2)

    def forward(self, x, t, conditional_image=None, conditional_vector_embedding=None):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if self.num_patches > 1:
            x = self.to_patches(x, patch_size=self.num_patches)
        x = self.in_layer(x)
        if self.use_conditional_image and conditional_image != None:
            conditional_image = self.to_patches(conditional_image, patch_size=self.num_patches)
            x = self.in_layer_conditional_img(x, conditional_image)
        if self.use_conditional_embedding and conditional_vector_embedding != None:
            conditional_vector_embedding = self.embedding_reshaping(conditional_vector_embedding)
            x = self.in_layer_conditional_vector(x, conditional_vector_embedding)

        # Down
        x_down_list = []
        for i in range(self.num_levels):
            x_down_list.append(x)
            conv = self.down_conv_layers[i]
            x = conv(x, t)
            x = self.activation_func(x)
            if  self.use_conditional_image and conditional_image != None:
                conditional = self.down_conditional_img_layers[i]
                x = conditional(x, conditional_image)
                x = self.activation_func(x)
            if  self.use_conditional_embedding and conditional_vector_embedding != None:
                conditional = self.down_conditional_txt_layers[i]
                x = conditional(x, conditional_vector_embedding)
                x = self.activation_func(x)
            if self.use_self_attention:
                att = self.down_att_layers[i]
                x = att(x)
                x = self.activation_func(x)

        for middle_layer in self.middle_layers:
            x = middle_layer(x)
            x = self.activation_func(x)

        for i in range(self.num_levels):
            conv = self.up_conv_layers[i]
            x_skip = x_down_list[ self.num_levels-1 - i ]
            x = conv(x, x_skip, t)
            x = self.activation_func(x)
            if  self.use_conditional_image and conditional_image != None:
                conditional = self.up_conditional_img_layers[i]
                x = conditional(x, conditional_image)
                x = self.activation_func(x)
            if  self.use_conditional_embedding and conditional_vector_embedding != None:
                conditional = self.up_conditional_txt_layers[i]
                x = conditional(x, conditional_vector_embedding)
                x = self.activation_func(x)
            if self.use_self_attention:
                att = self.up_att_layers[i]
                x = att(x)
                x = self.activation_func(x)

        if  self.use_conditional_image and conditional_image != None:
            x = self.out_layer_conditional_img(x, conditional_image)
        if  self.use_conditional_embedding and conditional_vector_embedding != None:
            x = self.out_layer_conditional_vector(x, conditional_vector_embedding)
        output = self.out_layer(x)
        if self.num_patches > 1:
            output = self.from_patches(output, patch_size=self.num_patches)
        if self.output_activation_func is not None:
            output = self.output_activation_func(output)
        return output