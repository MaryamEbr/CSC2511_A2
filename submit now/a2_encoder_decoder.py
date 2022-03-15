'''
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall
Updated by: Raeid Saqur <raeidsaqur@cs.toronto.edu>

All of the files in this directory and all subdirectories are:
Copyright (c) 2022 University of Toronto
'''

'''Concrete implementations of abstract base classes.

You don't need anything more than what's been imported here
'''

import torch
from typing import Optional, Union, Tuple, Type, Set

from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase


# All docstrings are omitted in this file for simplicity. So please read
# a2_abcs.py carefully so that you can have a solid understanding of the
# structure of the assignment.

class Encoder(EncoderBase):

    def init_submodules(self):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.rnn, self.embedding
        # 2. You will need the following object attributes:
        #   self.source_vocab_size, self.word_embedding_size,
        #   self.pad_id, self.dropout, self.cell_type,
        #   self.hidden_state_size, self.num_hidden_layers.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules: torch.nn.{LSTM, GRU, RNN, Embedding}

        self.embedding = torch.nn.Embedding(num_embeddings=self.source_vocab_size,
                                            embedding_dim=self.word_embedding_size,
                                            padding_idx=self.pad_id)
        # print("##### NEW NEW NEW (ENCODER) in init padding_idx ", self.pad_id)
        if self.cell_type == 'rnn':
            self.rnn = torch.nn.RNN(input_size=self.word_embedding_size, hidden_size=self.hidden_state_size,
                                    num_layers=self.num_hidden_layers, dropout=self.dropout, bidirectional=True)

        elif self.cell_type == 'gru':
            self.rnn = torch.nn.GRU(input_size=self.word_embedding_size, hidden_size=self.hidden_state_size,
                                    num_layers=self.num_hidden_layers, dropout=self.dropout, bidirectional=True)

        elif self.cell_type == 'lstm':
            self.rnn = torch.nn.LSTM(input_size=self.word_embedding_size, hidden_size=self.hidden_state_size,
                                     num_layers=self.num_hidden_layers, dropout=self.dropout, bidirectional=True)

    def forward_pass(
            self,
            F: torch.LongTensor,
            F_lens: torch.LongTensor,
            h_pad: float = 0.) -> torch.FloatTensor:
        # Recall:
        #   F is shape (S, M)
        #   F_lens is of shape (M,)
        #   h_pad is a float
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   input seq -> |embedding| -> embedded seq -> |rnn| -> seq hidden
        # 2. You will need to use the following methods:
        #   self.get_all_rnn_inputs, self.get_all_hidden_states

        # print("*** in encoder  ")
        # print("F shape ", F.shape)
        # print("F_Lens ", F_lens)
        # print("h_pad ", h_pad)
        # print("testing this padding thingy  ")
        # print("pad id ", self.pad_id)
        # print("F, m=0 ", F[:,0])
        # print("__________________________")
        # print("F, m=1 ", F[:,1])
        # print("__________________________")
        # print("F, m=2 ", F[:,2])
        # print("__________________________")
        # print("F, m=3 ", F[:,3])
        # print("__________________________")
        # print("F, m=4 ", F[:,4])
        # print("__________________________")


        x = self.get_all_rnn_inputs(F)
        hid = self.get_all_hidden_states(x, F_lens, h_pad)
        return hid

    def get_all_rnn_inputs(self, F: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   F is shape (S, M)
        #   x (output) is shape (S, M, I)

        # print("*** in get rnn input")
        x = self.embedding(F)
        # print("embed size: ", x.shape)

        return x


    def get_all_hidden_states(
            self,
            x: torch.FloatTensor,
            F_lens: torch.LongTensor,
            h_pad: float) -> torch.FloatTensor:
        # Recall:
        #   x is of shape (S, M, I)
        #   F_lens is of shape (M,)
        #   h_pad is a float
        #   h (output) is of shape (S, M, 2 * H)
        #
        # Hint:
        #   relevant pytorch modules:
        #   torch.nn.utils.rnn.{pad_packed,pack_padded}_sequence

        # print("*** in get all hidden states ")
        # print("x shape (embed result): ", x.shape)


        ### the encoder doesn't process the padding
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, F_lens.to('cpu'), enforce_sorted=False)

        out, hidden = self.rnn(packed)

        # print("after RNN ", self.cell_type)
        seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(sequence=out, padding_value=h_pad)
        # print("seq unpacked shape ",seq_unpacked.shape)

        return seq_unpacked


class DecoderWithoutAttention(DecoderBase):
    '''A recurrent decoder without attention'''

    def init_submodules(self):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.embedding, self.cell, self.ff
        # 2. You will need the following object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}

        self.embedding = torch.nn.Embedding(num_embeddings=self.target_vocab_size,
                                            embedding_dim=self.word_embedding_size,
                                            padding_idx=self.pad_id)
        # print("##### NEW NEW NEW (DECODER) in init padding_idx ", self.pad_id)
        self.ff = torch.nn.Linear(in_features=self.hidden_state_size, out_features=self.target_vocab_size)

        if self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(input_size=self.word_embedding_size, hidden_size=self.hidden_state_size)

        elif self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(input_size=self.word_embedding_size, hidden_size=self.hidden_state_size)

        elif self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(input_size=self.word_embedding_size, hidden_size=self.hidden_state_size)

    def forward_pass(
            self,
            E_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> Tuple[
        torch.FloatTensor, Union[
            torch.FloatTensor,
            Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Recall:
        #   E_tm1 is of shape (M,)
        #   htilde_tm1 is of shape (M, 2 * H)
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   logits_t (output) is of shape (M, V)
        #   htilde_t (output) is of same shape as htilde_tm1
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   encoded hidden -> |embedding| -> embedded hidden -> |rnn| ->
        #   decoded hidden -> |output layer| -> output logits
        # 2. You will need to use the following methods:
        #   self.get_current_rnn_input, self.get_current_hidden_state,
        #   self.get_current_logits
        # 3. You can assume that htilde_tm1 is not empty. I.e., the hidden state
        #   is either initialized, or t > 1.
        # 4. The output of an LSTM cell is a tuple (h, c), but a GRU cell or an
        #   RNN cell will only output h.


        # print("^^^ YAY, here in decoder forward pass  ")
        # print("E_tm1 shape ", E_tm1.shape)
        # print("E_tm1  ", E_tm1)
        # print("htilde_tm1 shape ", htilde_tm1.shape)
        # print("htilde_tm1 ", htilde_tm1)
        # print("h shape ", h.shape)
        # print("F_lens ", F_lens)

        xtilde_t = self.get_current_rnn_input(E_tm1, htilde_tm1, h, F_lens)
        # print("^^^ in forward pass again xtilde shape", xtilde_t.shape)

        htilde_t = self.get_current_hidden_state(xtilde_t, htilde_tm1)

        # Since the output of LTSM cell is a tuple
        if self.cell_type == 'lstm':
            logits_t = self.get_current_logits(htilde_t[0])
        else:
            logits_t = self.get_current_logits(htilde_t)

        return logits_t, htilde_t

    def get_first_hidden_state(
            self,
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   htilde_tm1 (output) is of shape (M, 2 * H)
        #
        # Hint:
        # 1. Ensure it is derived from encoder hidden state that has processed
        # the entire sequence in each direction. You will need to:
        # - Populate indices [0: self.hidden_state_size // 2] with the hidden
        #   states of the encoder's forward direction at the highest index in
        #   time *before padding*
        # - Populate indices [self.hidden_state_size//2:self.hidden_state_size]
        #   with the hidden states of the encoder's backward direction at time
        #   t=0
        # 2. Relevant pytorch function: torch.cat

        # print("^^^ get_first_hidden_state")
        # print("h shape", h.shape)

        # ignore right padded h: h[F_lens[m]:, m]
        forward_states = h[F_lens - 1, torch.arange(F_lens.size(0)), : self.hidden_state_size // 2]
        # indeces are based on the hint
        backward_states = h[0, :, self.hidden_state_size // 2: self.hidden_state_size]
        htilde_0 = torch.cat([forward_states, backward_states], dim=1)
        # print("htilde_0 shape: ", htilde_0.shape)
        return htilde_0

    def get_current_rnn_input(
            self,
            E_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   E_tm1 is of shape (M,)
        #   htilde_tm1 is of shape (M, 2 * H) or a tuple of two of those (LSTM)
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   xtilde_t (output) is of shape (M, Itilde)


        # print("^^^ get_current_rnn_input")
        # print("E_tm1 ", E_tm1)
        xtilde_t = self.embedding(E_tm1)

        # print("after embed, xtilde shape ", xtilde_t.shape)


        return xtilde_t

    def get_current_hidden_state(
            self,
            xtilde_t: torch.FloatTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]]) -> Union[
        torch.FloatTensor,
        Tuple[torch.FloatTensor, torch.FloatTensor]]:
        # Recall:
        #   xtilde_t is of shape (M, Itilde)
        #   htilde_tm1 is of shape (M, 2 * H) or a tuple of two of those (LSTM)
        #   htilde_t (output) is of same shape as htilde_tm1


        # print("^^^ get_current_hidden_state")

        ### seperating LSTM becuase it's a tuple
        if self.cell_type == 'lstm':
            htilde_tm1 = [htilde_tm1[0][:, :self.hidden_state_size], htilde_tm1[1][:, :self.hidden_state_size]]
        else:
            htilde_tm1 = htilde_tm1[:, :self.hidden_state_size]

        htilde_t = self.cell(xtilde_t, htilde_tm1)
        return htilde_t

    def get_current_logits(self, htilde_t: torch.FloatTensor) -> torch.FloatTensor:
        # Recall:
        #   htilde_t is of shape (M, 2 * H), even for LSTM (cell state discarded)
        #   logits_t (output) is of shape (M, V)


        return self.ff(htilde_t)


class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):
        # Hints:
        # 1. Same as the case without attention, you must initialize the
        #   following submodules: self.embedding, self.cell, self.ff
        # 2. You will need the following object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        # 5. The implementation of this function should be different from
        #   DecoderWithoutAttention.init_submodules.
        self.embedding = torch.nn.Embedding(num_embeddings=self.target_vocab_size,
                                            embedding_dim=self.word_embedding_size,
                                            padding_idx=self.pad_id)

        self.ff = torch.nn.Linear(in_features=self.hidden_state_size, out_features=self.target_vocab_size)

        ### this is the different part, xtilde is different when we use attention
        ### the size of hidden states are added to the word_embedding_size
        if self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(input_size=self.hidden_state_size + self.word_embedding_size,
                                        hidden_size=self.hidden_state_size)

        elif self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(input_size=self.hidden_state_size + self.word_embedding_size,
                                        hidden_size=self.hidden_state_size)

        elif self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(input_size=self.hidden_state_size + self.word_embedding_size,
                                         hidden_size=self.hidden_state_size)

    def get_first_hidden_state(
            self,
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # Hint: For this time, the hidden states should be initialized to zeros.
        htilde_0 = torch.zeros_like(h[0])

        # htilde_0 = torch.zeros([h.shape[1], self.hidden_state_size])
        # print("^^^ ATTN ^^^ get_first_hidden_state")
        # print("htilde_0  shape", htilde_0.shape)
        # print("hhh other shape ", h[1].shape)
        # print("compare ",(torch.zeros_like(h[0]) ==  torch.zeros([h.shape[1], self.hidden_state_size])))

        return htilde_0

    def get_current_rnn_input(
            self,
            E_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:

        # print("^^^ ATTN ^^^ get_current_rnn_input")
        # print("^^^ before attend ")
        # print("htilde_tm1 shape ", htilde_tm1.shape)
        # print("h shape ", h.shape)
        # print("Flens ", F_lens)

        # Hint: Use attend() for c_t


        ### separate LSTM because it's a tuple we only need hidden states
        if (self.cell_type == 'lstm'):
            htilde_tm1 = htilde_tm1[0]

        c_t = self.attend(htilde_tm1, h, F_lens)
        xtilde_t = self.embedding(E_tm1)

        # print("^^^ last in current rnn input, the xtilde after attend ")
        # print("xtilde (prev) shape ", xtilde_t.shape)
        # print("c_t shape  ", c_t.shape)
        # print("cat shape ", torch.cat([xtilde_t, c_t], dim=1).shape)
        return torch.cat([xtilde_t, c_t], dim=1)

    def attend(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        '''The attention mechanism. Calculate the context vector c_t.

        Parameters
        ----------
        htilde_t : torch.FloatTensor or tuple
            Like `htilde_tm1` (either a float tensor or a pair of float
            tensors), but matching the current hidden state.
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[F_lens[m]:, m]``
            should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        c_t : torch.FloatTensor
            A float tensor of shape ``(M, self.hidden_state_size)``. The
            context vector c_t is the product of weights alpha_t and h.

        Hint: Use get_attention_weights() to calculate alpha_t.
        '''



        alpha_t = self.get_attention_weights(htilde_t, h, F_lens)
        # print("^^^ back at ATTEND ")
        # print("alpha_t shape (S,M) ", alpha_t.shape)
        # print("h shape (S, M, hidden) ", h.shape)


        ### before mul, h and alpha should have same dimension.
        ### h is (S, M, hidden) and alpha is (S, M)
        ### to add another dimension to alpha:
        alpha_t = alpha_t.unsqueeze(2)

        #### ???? is it ok to change alpha shape???

        c_t = torch.sum(torch.mul(alpha_t, h), dim=0)
        # print("c_t shape, last in ATTEND (M, hidden) ", c_t.shape)
        return c_t

    def get_attention_weights(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_attention_scores()
        # alpha_t (output) is of shape (S, M)

        # print("^^^ in get_attention_weights")
        e_t = self.get_attention_scores(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= F_lens.to(h.device)  # (S, M)
        e_t = e_t.masked_fill(pad_mask, -float('inf'))
        # print("^^^ IN SCORE checking the final shape of alpha ", torch.nn.functional.softmax(e_t, 0).shape)
        return torch.nn.functional.softmax(e_t, 0)

    def get_attention_scores(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor) -> torch.FloatTensor:
        # Recall:
        #   htilde_t is of shape (M, 2 * H)
        #   h is of shape (S, M, 2 * H)
        #   e_t (output) is of shape (S, M)
        #
        # Hint:
        # Relevant pytorch function: torch.nn.functional.cosine_similarity

        # print("^^^ in get_attention_scores")

        htilde_t = htilde_t.unsqueeze(0)

        e_t = torch.nn.CosineSimilarity(dim=2)(h, htilde_t)
        # print("e_t shape (S, M) ", e_t.shape)
        return e_t


class DecoderWithMultiHeadAttention(DecoderWithAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.W is not None, 'initialize W!'
        assert self.Wtilde is not None, 'initialize Wtilde!'
        assert self.Q is not None, 'initialize Q!'

    def init_submodules(self):
        super().init_submodules()  # Do not change this line

        # Hints:
        # 1. The above line should ensure self.ff, self.embedding, self.cell are
        #    initialized
        # 2. You need to initialize the following submodules:
        #       self.W, self.Wtilde, self.Q
        # 3. You will need the following object attributes:
        #       self.hidden_state_size
        # 4. self.W, self.Wtilde, and self.Q should process all heads at once. They
        #    should not be lists!
        # 5. You do *NOT* need self.heads at this point
        # 6. Relevant pytorch module: torch.nn.Linear (note: set bias=False!)

        self.W = torch.nn.Linear(in_features=self.hidden_state_size, out_features=self.hidden_state_size, bias=False)
        self.Wtilde = torch.nn.Linear(in_features=self.hidden_state_size, out_features=self.hidden_state_size,
                                      bias=False)
        self.Q = torch.nn.Linear(in_features=self.hidden_state_size, out_features=self.hidden_state_size, bias=False)

    def attend(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # Hints:
        # 1. You can use super().attend to call for the regular attention
        #   function.
        # 2. Relevant pytorch function:
        #   tensor().view, tensor().repeat_interleave
        # 3. Fun fact:
        #   tensor([1,2,3,4]).repeat(2) will output tensor([1,2,3,4,1,2,3,4]).
        #   tensor([1,2,3,4]).repeat_interleave(2) will output
        #   tensor([1,1,2,2,3,3,4,4]), just like numpy.repeat.
        # 4. You *WILL* need self.heads at this point

        number_of_parts = self.hidden_state_size//self.heads
        S = h.shape[0]
        M = h.shape[1]


        h = self.W(h)
        htilde_t = self.Wtilde(htilde_t)

        ### reshape based on heads
        h = h.view(S, -1, number_of_parts)
        htilde_t = htilde_t.view(-1, number_of_parts)

        c_t_n = super().attend(htilde_t, h, F_lens.repeat_interleave(repeats=self.heads))

        ### reshape back
        c_t_n = c_t_n.view(M, self.hidden_state_size)
        # print("c_t_n shape after ",c_t_n.shape)

        return self.Q(c_t_n)


class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(
            self,
            encoder_class: Type[EncoderBase],
            decoder_class: Type[DecoderBase]):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.encoder, self.decoder
        # 2. encoder_class and decoder_class inherit from EncoderBase and
        #   DecoderBase, respectively.
        # 3. You will need the following object attributes:
        #   self.source_vocab_size, self.source_pad_id,
        #   self.word_embedding_size, self.encoder_num_hidden_layers,
        #   self.encoder_hidden_size, self.encoder_dropout, self.cell_type,
        #   self.target_vocab_size, self.target_eos, self.heads
        # 4. Recall that self.target_eos doubles as the decoder pad id since we
        #   never need an embedding for it
        self.encoder = encoder_class(source_vocab_size=self.source_vocab_size,
                                     pad_id=self.source_pad_id,
                                     word_embedding_size=self.word_embedding_size,
                                     num_hidden_layers=self.encoder_num_hidden_layers,
                                     hidden_state_size=self.encoder_hidden_size,
                                     dropout=self.encoder_dropout,
                                     cell_type=self.cell_type)
        self.encoder.init_submodules()
        # print("#### ENCODER DECODER source pad id ", self.source_pad_id)
        # print("#### ENCODER DECODER target eos ", self.target_eos)
        self.decoder = decoder_class(target_vocab_size=self.target_vocab_size,
                                     pad_id=self.target_eos,
                                     word_embedding_size=self.word_embedding_size,
                                     hidden_state_size=self.encoder_hidden_size * 2,
                                     cell_type=self.cell_type,
                                     heads=self.heads)
        self.decoder.init_submodules()

    def get_logits_for_teacher_forcing(
            self,
            h: torch.FloatTensor,
            F_lens: torch.LongTensor,
            E: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   E is of shape (T, M)
        #   logits (output) is of shape (T - 1, M, Vo)
        #
        # Hints:
        # 1. Relevant pytorch modules: torch.{zero_like, stack}
        # 2. Recall an LSTM's cell state is always initialized to zero.
        # 3. Note logits sequence dimension is one shorter than E (why?)
        # size of logits are 1 shorter, it's probably because of eos? sos?


        logit_list = []
        h_tilde_tm1 = None
        for t in range(E.shape[0] - 1):
            logit, h_tilde_tm1 = self.decoder.forward(E[t], h_tilde_tm1, h, F_lens)
            logit_list.append(logit)
        logits = torch.stack(logit_list, dim=0)
        # print("^^^^ in teacher forcing logit shape  ", logits.shape)
        return logits


    def update_beam(
            self,
            htilde_t: torch.FloatTensor,
            b_tm1_1: torch.LongTensor,
            logpb_tm1: torch.FloatTensor,
            logpy_t: torch.FloatTensor) -> Tuple[
        torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        # perform the operations within the psuedo-code's loop in the
        # assignment.
        # You do not need to worry about which paths have finished, but DO NOT
        # re-normalize logpy_t.
        #
        # Recall
        #   htilde_t is of shape (M, K, 2 * H) or a tuple of two of those (LSTM)
        #   logpb_tm1 is of shape (M, K)
        #   b_tm1_1 is of shape (t, M, K)
        #   b_t_0 (first output) is of shape (M, K, 2 * H) or a tuple of two of
        #      those (LSTM)
        #   b_t_1 (second output) is of shape (t + 1, M, K)
        #   logpb_t (third output) is of shape (M, K)
        #
        # Hints:
        # 1. Relevant pytorch modules:
        #   torch.{flatten, topk, unsqueeze, expand_as, gather, cat}
        # 2. If you flatten a two-dimensional array of shape z of (A, B),
        #   then the element z[a, b] maps to z'[a*B + b]
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ FINALLY BEAM BEAM")
        # print("htilde_t: ",htilde_t.shape)
        # print("btm1_1: ", b_tm1_1.shape)
        # print("logpb_tm1: ", logpb_tm1.shape)
        # print("logpy_t: ", logpy_t.shape)

        ### logpb_t
        ext_paths = logpb_tm1.unsqueeze(-1) + logpy_t
        ext_paths = ext_paths.view((ext_paths.shape[0]), -1)
        logpb_t, top_k = torch.topk(ext_paths, k=self.beam_width, dim=-1, largest=True, sorted=True)

        # print("Result of topk  ", logpb_t, " -v ", top_k)

        ### chosen
        paths = torch.div(top_k, logpy_t.shape[-1],  rounding_mode='floor')
        v_indecies = torch.remainder(top_k, logpy_t.shape[-1])

        ### path seq, chech the shapes at the end??
        b_t_1 = torch.cat([torch.gather(b_tm1_1, dim=2, index=paths.unsqueeze(0).expand_as(b_tm1_1)), v_indecies.unsqueeze(0)],
                          dim=0)

        ### seperate LSTM
        if self.decoder.cell_type == "lstm":
            b_t_0 = (torch.gather(htilde_t[0], dim=1, index=torch.unsqueeze(paths, dim=2).expand_as(htilde_t[0])),
                     torch.gather(htilde_t[1], dim=1, index=torch.unsqueeze(paths, dim=2).expand_as(htilde_t[1])))
        else:
            b_t_0 = torch.gather(htilde_t, dim=1, index=torch.unsqueeze(paths, dim=2).expand_as(htilde_t))

        # print("b_t_0 shape ", b_t_0.shape)
        # print("b_t_1 shape ", b_t_1.shape)
        # print("logpb_t shape ", logpb_t.shape)
        return b_t_0, b_t_1, logpb_t
