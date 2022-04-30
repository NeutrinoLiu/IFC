"""
https://github.com/facebookresearch/detr
"""
import copy
from logging.config import valid_ident
from signal import default_int_handler
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from ..ops.modules import MSDeformAttn

class IFCTransformer(nn.Module):

    def __init__(self, num_frames, d_model=256, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3,
                 num_memory_bus=8, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,
                 deformable=0,
                 MLPEncoder=0):
        super().__init__()

        self.num_frames = num_frames
        self.num_memory_bus = num_memory_bus
        self.MLPEncoder = MLPEncoder

        if MLPEncoder:
            encoder_layer = MLPMixerEncoderLayer(d_model, MLPEncoder, num_memory_bus, dropout, activation)
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, 
                                                deformable)
        memory_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, 
                                                deformable=0) # do not use deformable for memory
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = IFCEncoder(num_frames, encoder_layer, memory_layer, num_encoder_layers, encoder_norm)
                                    

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.clip_decoder = IFCDecoder(decoder_layer, num_decoder_layers, num_frames, decoder_norm,
                                        return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.memory_bus = torch.nn.Parameter(torch.randn(num_memory_bus, d_model))
        self.memory_pos = torch.nn.Parameter(torch.randn(num_memory_bus, d_model))
        if num_memory_bus:
            nn.init.kaiming_normal_(self.memory_bus, mode="fan_out", nonlinearity="relu")
            nn.init.kaiming_normal_(self.memory_pos, mode="fan_out", nonlinearity="relu")

        self.return_intermediate_dec = return_intermediate_dec

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def pad_zero(self, x, pad, dim=0):
        if x is None:
            return None
        pad_shape = list(x.shape)
        pad_shape[dim] = pad
        return torch.cat((x, x.new_zeros(pad_shape)), dim=dim)

    def forward(self, src, mask, query_embed, pos_embed, is_train):
        # prepare for enc-dec
        bs = src.shape[0] // self.num_frames if is_train else 1
        t = src.shape[0] // bs
        _, c, h, w = src.shape
        all_shape = {'h':h, 'w':w, 'bs':bs, 't':t}

        memory_bus = self.memory_bus
        memory_pos = self.memory_pos

        # encoder
        src = src.view(bs*t, c, h*w).permute(2, 0, 1)               # HW, BT, C
        frame_pos = pos_embed.view(bs*t, c, h*w).permute(2, 0, 1)   # HW, BT, C
        frame_mask = mask.view(bs*t, h*w)                           # BT, HW

        src, memory_bus = self.encoder(src, memory_bus, memory_pos, src_key_padding_mask=frame_mask, pos=frame_pos, is_train=is_train, all_shape=all_shape)

        # decoder
        dec_src = src.view(h*w, bs, t, c).permute(2, 0, 1, 3).flatten(0,1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)     # Q, B, C
        tgt = torch.zeros_like(query_embed)

        dec_pos = pos_embed.view(bs, t, c, h*w).permute(1, 3, 0, 2).flatten(0,1)
        dec_mask = mask.view(bs, t*h*w)                             # B, THW

        clip_hs = self.clip_decoder(tgt, dec_src, memory_bus, memory_pos, memory_key_padding_mask=dec_mask,
                                    pos=dec_pos, query_pos=query_embed, is_train=is_train)

        ret_memory = src.permute(1,2,0).reshape(bs*t, c, h, w)

        return clip_hs, ret_memory


class IFCEncoder(nn.Module):

    def __init__(self, num_frames, encoder_layer, memory_layer, num_layers, norm=None):
        super().__init__()
        self.num_frames = num_frames
        self.num_layers = num_layers
        self.enc_layers = _get_clones(encoder_layer, num_layers)
        self.bus_layers = _get_clones(memory_layer, num_layers)
        norm = [copy.deepcopy(norm) for i in range(2)]
        self.out_norm, self.bus_norm = norm

    @staticmethod
    def pad_zero(x, pad, dim=0):
        if pad == 0:
            return x
        if x is None:
            return None
        pad_shape = list(x.shape)
        pad_shape[dim] = pad
        return torch.cat((x, x.new_zeros(pad_shape)), dim=dim)

    @staticmethod
    def pad_one(x, pad, dim=0):
        if pad == 0:
            return x
        if x is None:
            return None
        pad_shape = list(x.shape)
        pad_shape[dim] = pad
        return torch.cat((x, x.new_ones(pad_shape)), dim=dim)

    def forward(self, src, memory_bus, memory_pos,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                is_train: bool = True,
                all_shape = None):
        bs = src.shape[1] // self.num_frames if is_train else 1
        t = src.shape[1] // bs
        hw, _, c = src.shape
        M = len(memory_bus)
        all_shape['M'] = M
        h = all_shape['h']
        w = all_shape['w']
        # print("[shapes]", all_shape)
        # cs839: hence deformable attention is a 2d attention, HxW
        # but IFC MHAtt is flattened 1d attention, HW+M
        # we will fill the memory line to a full width of feature map
        # so it looks like a 2d image, for example:
        #   $$$$$$
        #   $$$$$$
        #   **oooo
        # $-feature *-memory o-dummy extension

        if M > w:
            raise ValueError(f"Too larget the memory bus size {M} than the width of feature: {w}")

        memory_pos = self.pad_zero(memory_pos, w-M)
        memory_pos = memory_pos[:, None, :].repeat(1, bs*t, 1)
        pos = torch.cat((pos, memory_pos))

        memory_bus = self.pad_zero(memory_bus, w-M)
        memory_bus = memory_bus[:, None, :].repeat(1, bs*t, 1)

        # cs839
        # now our image has height of h+1 with the extraline memory bus and its dummy extension
        # no need for the dummy filling, we note the padding mask bit as True at that position
        all_shape['h'] += 1
        mask = self.pad_zero(mask, M, dim=1)
        mask = self.pad_one(mask, w-M, dim=1)
        src_key_padding_mask = self.pad_zero(src_key_padding_mask, M, dim=1)
        src_key_padding_mask = self.pad_one(src_key_padding_mask, w-M, dim=1)
        
        output = src

        for layer_idx in range(self.num_layers):
            
            if not isinstance(self.enc_layers[0], MLPMixerEncoderLayer):
                output = torch.cat((output, memory_bus))
                output = self.enc_layers[layer_idx](output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, all_shape=all_shape)
                output, memory_bus = output[:hw, :, :], output[hw:hw+M, :, :]
                memory_bus = memory_bus.view(M, bs, t, c).permute(2,1,0,3).flatten(1,2) # TxBMxC
                memory_bus = self.bus_layers[layer_idx](memory_bus)
                memory_bus = memory_bus.view(t, bs, M, c).permute(2,1,0,3).flatten(1,2) # MxBTxC
                memory_bus = self.pad_zero(memory_bus, w-M)
            else:
                output = self.enc_layers[layer_idx](output[:hw, :, :], memory_bus[:M, :, :]) 
                output, memory_bus = output[:hw, :, :], output[-M:, :, :]
                memory_bus = memory_bus.view(M, bs, t, c).permute(2,1,0,3).flatten(1,2) # TxBMxC
                memory_bus = self.bus_layers[layer_idx](memory_bus)
                memory_bus = memory_bus.view(t, bs, M, c).permute(2,1,0,3).flatten(1,2) # MxBTxC
            
            

        # do not forget to remove the dummy extension
        memory_bus = memory_bus[:M]

        if self.out_norm is not None:
            output = self.out_norm(output)
        if self.bus_norm is not None:
            memory_bus = self.bus_norm(memory_bus)

        return output, memory_bus


class IFCDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, num_frames, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.num_frames = num_frames
        self.norm = norm
        self.return_intermediate = return_intermediate

    def pad_zero(self, x, pad, dim=0):
        if x is None:
            return None
        pad_shape = list(x.shape)
        pad_shape[dim] = pad
        return torch.cat((x, x.new_zeros(pad_shape)), dim=dim)

    def forward(self, tgt, memory, memory_bus, memory_pos,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                is_train: bool = True):
        output = tgt

        return_intermediate = (self.return_intermediate and is_train)
        intermediate = []

        M, bt, c = memory_bus.shape
        bs = bt // self.num_frames if is_train else 1
        t = bt // bs

        memory_bus = memory_bus.view(M, bs, t, c).permute(2,0,1,3).flatten(0,1) # TMxBxC
        memory = torch.cat((memory, memory_bus))

        memory_pos = memory_pos[None, :, None, :].repeat(t,1,bs,1).flatten(0,1) # TMxBxC
        pos = torch.cat((pos, memory_pos))

        memory_key_padding_mask = self.pad_zero(memory_key_padding_mask, t*M, dim=1) # B, THW

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

class MLP(nn.Module):
    def __init__(self, d_model, d_hidden=None, activation='gelu'):
        super().__init__()
        if not d_hidden:
            d_hidden = d_model
        self.l1 = nn.Linear(d_model, d_hidden)
        self.acti = _get_activation_fn(activation)
        self.l2 = nn.Linear(d_hidden, d_model)
        nn.init.kaiming_normal_(self.l1.weight, mode="fan_out", nonlinearity='relu')
        nn.init.kaiming_normal_(self.l2.weight, mode="fan_out", nonlinearity='relu')

    def forward(self, input):
        ret = self.l1(input)
        ret = self.acti(ret)
        ret = self.l2(ret)
        return ret

class MLPMixerEncoderLayer(nn.Module):
    def __init__(self, d_model, max_length=512, memory_bus=8, dropout=0.1, activation="relu"):
        super().__init__()

        self.d_model = d_model
        self.max_length = max_length
        self.memory_bus = memory_bus

        self.tokenMixer = MLP(max_length+memory_bus, d_hidden=15*27+8)
        self.channelMixer = MLP(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def pad_zero(self, x, pad, dim=0):
        if x is None:
            return None
        pad_shape = list(x.shape)
        pad_shape[dim] = pad
        return torch.cat((x, x.new_zeros(pad_shape)), dim=dim)

    def forward(self, src, memory): # src-(L[hw+m], N, C), mask-(N, L[hw+m]), no need for pos in MLP Mixer
        
        NLC2LNC = LNC2NLC = lambda src: src.permute(1,0,2)
        # assert size
        src = LNC2NLC(src)
        memory = LNC2NLC(memory)
        N = src.shape[0]
        L = src.shape[1]
        C = src.shape[2]
        assert C == self.d_model
        assert self.memory_bus == memory.shape[1]

        if L < self.max_length:
            src = self.pad_zero(src, self.max_length-L, dim=1)
            L = self.max_length
        elif L > self.max_length:
            raise ValueError("too large the input image!")
        # print(src.shape, memory.shape)
        src = torch.cat((src, memory), dim=1)
        L += self.memory_bus
        # print(src.shape)
        # print(N, L, C)
        src2 = self.norm1(src).permute(0, 2, 1).flatten(0,1) # N, L ,C -> N, C, L
        src2 = self.tokenMixer(src2)
        src2 = src2.view(N, C, L).permute(0, 2, 1) # N, C, L -> N, L ,C
        src = src + self.dropout1(src2)

        src2 = self.norm2(src).flatten(0,1)
        src2 = self.channelMixer(src2)
        src2 = src2.view(N, L, C)
        src = src + self.dropout2(src2)
        
        return NLC2LNC(src)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 deformable=0):
        super().__init__()
        self.deformable = deformable
        if not deformable:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            self.self_attn = MSDeformAttn(d_model, 1, nhead, deformable) 
            # cs839
            # para deformable also indicates number of points
            # we only use last layer of backbone instead of multi scale to reduce computation
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                        torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    @staticmethod
    def get_valid_ratio(mask, all_shape):
        H = all_shape['h']
        W = all_shape['w']
        mask = mask.view(-1, H, W)
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     all_shape=None):
        q = k = self.with_pos_embed(src, pos)

        if not self.deformable:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask)[0]
                            # nn.MHA will output (output, weights), only former is needed
        else:
            # cs839 different interface for deformable attention
            if all_shape == None:
                raise ValueError("frame_shape is not provided for forward")
            spatial_shapes = torch.as_tensor([(all_shape['h'], all_shape['w'])], dtype=torch.long, device=src.device)
            level_start_index = torch.as_tensor([0, src_key_padding_mask.shape[1]], dtype=torch.long, device=src.device)
            
            valid_ratios = torch.stack([self.get_valid_ratio(src_key_padding_mask, all_shape)],1)
            reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
            
            # cs839
            # util function used by deformable-attention
            NLC2LNC = LNC2NLC = lambda src: src.permute(1,0,2)

            src2 = NLC2LNC(
                self.self_attn(
                    LNC2NLC(q),
                    reference_points,
                    LNC2NLC(src),
                    spatial_shapes,
                    level_start_index,
                    src_key_padding_mask)
                )

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    all_shape = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)

        if not self.deformable:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask)[0]
                            # nn.MHA will output (output, weights), only former is needed
        else:
            # cs839 different interface for deformable attention
            if all_shape == None:
                raise ValueError("frame_shape is not provided for forward")
            spatial_shapes = [(all_shape['h'], all_shape['w'])]
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src.device)
            level_start_index = [0, src_key_padding_mask.shape[1]]
            
            valid_ratios = self.get_valid_ratio(src_key_padding_mask, all_shape)
            reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
            
            # cs839
            # util function used by deformable-attention
            NLC2LNC = LNC2NLC = lambda src: src.permute(1,0,2)

            src2 = NLC2LNC(
                self.self_attn(
                    LNC2NLC(q),
                    reference_points,
                    LNC2NLC(src),
                    spatial_shapes,
                    level_start_index,
                    src_key_padding_mask)
                )

        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                all_shape = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, all_shape)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, all_shape)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
