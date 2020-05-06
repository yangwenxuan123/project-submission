
import torch
import torch.nn as nn
import torch.nn.functional as F


class NMTModel(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, hidden_dim,  vocab,device, bidirectional = True):
        super(NMTModel, self).__init__()
        self.vocab = vocab
        self.device = device
        self.embedding = nn.Embedding(vocab_dim, embedding_dim, padding_idx=vocab.stoi['<pad>'])
        self.encoder = Encoder(vocab.stoi, self.embedding, hidden_size=hidden_dim, bidirectional=bidirectional, dropout=0.2, device= self.device)
        self.eacher_forcing = False
        self.decoder = Decoder(hidden_size=hidden_dim, embedding=self.embedding, vocab2id=vocab.stoi, device = self.device)

    def forward(self, inputs, outputs, teacher_forcing= False):

        encoder_outputs = self.encoder(inputs)
        dec_input = torch.cat(
            (torch.tensor([self.vocab.stoi['<bos>']] * inputs.shape[0]).unsqueeze(1).to(self.device), outputs), dim=1)
        outputs = self.decoder(0, encoder_outputs, 0, 0, dec_input, teacher_forcing)
        return outputs

    # def inference(self, inputs, outputs):
    #     encoder_outputs = self.encoder(inputs)
    #     outputs = self.decoder(0, encoder_outputs, 0, 0, outputs)
    #     return outputs





class Encoder(nn.Module):
    def __init__(self, vocab2id, embedding, hidden_size,
                 bidirectional, dropout, device):
        super().__init__()
        embed_dim = embedding.embedding_dim
        self.embed_dim = embed_dim
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = 1
        self.pad_idx = vocab2id['<pad>']
        self.dropout = dropout
        self.device = device
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            bidirectional=bidirectional,
            batch_first=True)

    def forward(self, inputs):
        """
        :param src_dict:
        :return:
        """
        src_tokens = inputs
        # src_lengths = src_dict
        batch_size = len(src_tokens)
        src_embed = self.embedding(src_tokens)
        src_embed = F.dropout(src_embed, p=self.dropout, training=self.training)

        total_length = src_embed.size(1)
        # packed_src_embed = nn.utils.rnn.pack_padded_sequence(src_embed,
        #                                                      src_lengths,
        #                                                      batch_first=True,
        #                                                      enforce_sorted=False)
        state_size = [ self.num_layers,batch_size, self.hidden_size]
        if self.bidirectional:
            state_size[0] *= 2
        h0 = src_embed.new_zeros(state_size)
        c0 = src_embed.new_zeros(state_size)
        hidden_states, (final_hiddens, final_cells) = self.lstm(src_embed, (h0, c0))
        # hidden_states, _ = nn.utils.rnn.pad_packed_sequence(hidden_states,
        #                                                     padding_value=self.pad_idx,
        #                                                     batch_first=True,
        #                                                     total_length=total_length)
        encoder_padding_mask = src_tokens.eq(self.pad_idx)
        if self.bidirectional:
            final_hiddens = torch.cat((final_hiddens[0], final_hiddens[1]), dim=1).unsqueeze(0)
            final_cells = torch.cat((final_cells[0], final_cells[1]), dim=1).unsqueeze(0)
        output = {'encoder_output': hidden_states,
                  'encoder_padding_mask': encoder_padding_mask,
                  'encoder_hidden': [final_hiddens, final_cells]}
        return output


class Decoder(nn.Module):
    def __init__(self, hidden_size, embedding, vocab2id, device):
        super(Decoder,self).__init__()
        self.hidden_size = hidden_size * 2
        self.embedding = embedding
        self.vocab2id = vocab2id
        self.device = device
        self.padidx = vocab2id['<pad>']
        self.auto_regressive = True

        self.lstm = nn.LSTM(
            input_size=embedding.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.attn_layer = Attention(self.hidden_size, self.hidden_size, 'concat')
        self.generate_proj = nn.Linear(self.hidden_size, len(vocab2id), bias=False)

    def forward(self, prev_output_tokens, encoder_output_dict, prev_context_state,
                prev_rnn_state, targets, regressive):

        if regressive:
            output = self.forward_rnn_auto_regressive(encoder_output_dict=encoder_output_dict,
                                                  prev_output_tokens=prev_output_tokens,
                                                  prev_rnn_state=prev_rnn_state,
                                                  prev_context_state=prev_context_state)
        else:
            output = self.forward_rnn_one_pass(encoder_output_dict, encoder_output_dict['encoder_hidden'], targets)


        return output

    def forward_rnn_one_pass(self, encoder_output_dict, encoder_hidden_state, input):
        encoder_output = encoder_output_dict['encoder_output']
        encoder_output_mask = encoder_output_dict['encoder_padding_mask']

        decoder_input = self.embedding(input[:, :-1])

        rnn_output, rnn_state = self.lstm(decoder_input, encoder_hidden_state)
        attn_output, attn_weights = self.attn_layer(rnn_output, encoder_output, encoder_output_mask)
        probs = torch.log_softmax(self.generate_proj(attn_output), dim=-1)
        return probs, attn_output, rnn_state

    def infer_rnn_auto_regressive(self, encoder_output_dict, vocab, length):
        encoder_output = encoder_output_dict['encoder_output']
        encoder_output_mask = encoder_output_dict['encoder_padding_mask']
        encoder_hidden_state = encoder_output_dict['encoder_hidden']
        decoder_input = self.embedding(torch.tensor([vocab.stoi['<pad>']]*encoder_output.shape[0]).to(self.device)).unsqueeze(1)
        result = torch.zeros((encoder_output.shape[0], length, 1)).to(self.device)
        for i in range(length):
            rnn_output, rnn_state = self.lstm(decoder_input, encoder_hidden_state)
            encoder_hidden_state = rnn_state
            attn_output, attn_weights = self.attn_layer(rnn_output, encoder_output, encoder_output_mask)
            probs = torch.argmax(torch.log_softmax(self.generate_proj(attn_output), dim= -1), dim= 2)
            result[:, i] = probs
            decoder_input = self.embedding(probs)
        return result



    def forward_rnn_auto_regressive(self, encoder_output_dict, prev_output_tokens,
                                    prev_rnn_state, prev_context_state):
        """
        :param encoder_output_dict:
        :param prev_output_tokens:
        :param prev_rnn_state:
        :param prev_context_state:
        :return:
        """
        encoder_output = encoder_output_dict['encoder_output']
        encoder_output_mask = encoder_output_dict['encoder_padding_mask']
        src_embed = self.embedding(prev_output_tokens)
        if self.input_feeding:
            prev_context_state = prev_context_state.unsqueeze(1)
            decoder_input = torch.cat([src_embed, prev_context_state], dim=2)
        else:
            decoder_input = src_embed
        rnn_output, rnn_state = self.lstm(decoder_input, prev_rnn_state)
        rnn_state = list(rnn_state)
        attn_output, attn_weights = self.attn_layer(rnn_output, encoder_output, encoder_output_mask)
        probs = torch.log_softmax(self.generate_proj(attn_output).squeeze(1), 1)
        return probs, attn_output.squeeze(1), rnn_state



class Attention(nn.Module):
    """
    implement attention mechanism
    """

    def __init__(self, input_dim, output_dim, score_mode='general'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.score_mode = score_mode
        if self.score_mode == 'general':
            self.attn = nn.Linear(self.output_dim, self.input_dim, bias=False)
        elif self.score_mode == 'concat':
            self.query_proj = nn.Linear(self.output_dim, self.output_dim, bias=False)
            self.key_proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
            self.concat_proj = nn.Linear(self.output_dim, 1)
        elif self.score_mode == 'dot':
            if self.input_dim != self.output_dim:
                raise ValueError('input and output dim must be equal when attention score mode is dot')
        else:
            raise ValueError('attention score mode error')
        self.output_proj = nn.Linear(self.input_dim + self.output_dim, self.output_dim)

    def score(self, query, key, encoder_padding_mask):
        """
        :param query:
        :param key:
        :param encoder_padding_mask:
        :return:
        """
        tgt_len = query.size(1)
        src_len = key.size(1)
        if self.score_mode == 'general':
            attn_weights = torch.bmm(self.attn(query), key.permute(0, 2, 1))
        elif self.score_mode == 'concat':
            query_w = self.query_proj(query.unsqueeze(2).repeat(1, 1, src_len, 1))
            key_w = self.key_proj(key.unsqueeze(1).repeat(1, tgt_len, 1, 1))
            score = torch.tanh(query_w + key_w)
            attn_weights = self.concat_proj(score)
            attn_weights = torch.squeeze(attn_weights, 3)
        elif self.score_mode == 'dot':
            attn_weights = torch.bmm(query, key.permute(0, 2, 1))

        # mask input padding to -Inf, they will be zero after softmax.
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.unsqueeze(1).repeat(1, tgt_len, 1)
            attn_weights.masked_fill_(encoder_padding_mask, float('-inf'))
        attn_weights = torch.softmax(attn_weights, 2)
        return attn_weights

    def forward(self, decoder_output, encoder_outputs, encoder_padding_mask):
        """
        :param decoder_output: B x tgt_dim
        :param encoder_outputs: B x L x src_dim
        :param encoder_padding_mask:
        :return:
        """
        attn_weights = self.score(decoder_output, encoder_outputs, encoder_padding_mask)
        context_embed = torch.bmm(attn_weights, encoder_outputs)
        attn_outputs = torch.tanh(self.output_proj(torch.cat([context_embed, decoder_output], dim=2)))
        return attn_outputs, attn_weights


