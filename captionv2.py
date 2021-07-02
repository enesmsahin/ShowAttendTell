import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from PIL import Image
from models import Encoder, EncoderWide, EncoderFPN, Decoder, Decoder2layer
import os
from imageio import imread
from PIL import Image
from utils import *
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

def caption_image_beam_search(encoder, decoder, cfgData, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    modelTypes = cfgData["Model Type"]

    decoderType = modelTypes["Decoder"]
    enable2LayerDecoder = modelTypes["Enable2LayerDecoder"]

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    # img = imresize(img, (256, 256))
    img = np.array(Image.fromarray(img).resize((256,256), resample=Image.BILINEAR))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    if enable2LayerDecoder:
        h_1, c_1 = decoder.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        h_2 = torch.zeros(size=(k, decoder.decoder_dim)).to(device) # initialize states of second layer of RNN with zeros
        c_2 = torch.zeros(size=(k, decoder.decoder_dim)).to(device)
    else:
        h, c = decoder.init_hidden_state(encoder_out)
    too_long = False

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        if enable2LayerDecoder:
            awe, alpha = decoder.attention(encoder_out, h_2)  # (s, encoder_dim), (s, num_pixels) # attention weighted encoding
            gate = decoder.sigmoid(decoder.f_beta(h_2))  # gating scalar, (s, encoder_dim)
        else:
            awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels) # attention weighted encoding
            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)
        awe = gate * awe

        if enable2LayerDecoder:
            if decoderType == "LSTM":
                batch_size = embeddings.shape[0]
                h_1, c_1 = decoder.decode_step_1(torch.cat([embeddings, awe], dim=1), (h_1[:batch_size], c_1[:batch_size]))  # (s, decoder_dim)
                h_2, c_2 = decoder.decode_step_2(h_1, (h_2, c_2))  # (s, decoder_dim)
            elif decoderType == "GRU":
                batch_size = embeddings.shape[0]
                h_1 = decoder.decode_step_1(torch.cat([embeddings, awe], dim=1), h_1[:batch_size])  # (s, decoder_dim)
                h_2 = decoder.decode_step_2(h_1, h_2)  # (s, decoder_dim)
            else:
                raise Exception("decoderType should be one of \"LSTM\", \"GRU\"!")
            scores = decoder.fc(h_2)  # (s, vocab_size)
            
        else:
            if decoderType == "LSTM":
                h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
            elif decoderType == "GRU":
                h = decoder.decode_step(torch.cat([embeddings, awe], dim=1), h)  # (s, decoder_dim)
            else:
                raise Exception("decoderType should be one of \"LSTM\", \"GRU\"!")
            scores = decoder.fc(h)  # (s, vocab_size)
        
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds.long()], alpha[prev_word_inds.long()].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        
        if enable2LayerDecoder:
            if decoderType == "LSTM":
                h_2 = h_2[prev_word_inds[incomplete_inds].long()]
                c_2 = c_2[prev_word_inds[incomplete_inds].long()]
            elif decoderType == "GRU":
                h_2 = h_2[prev_word_inds[incomplete_inds].long()]
            else:
                raise Exception("decoderType should be one of \"LSTM\", \"GRU\"!")                
        else:
            if decoderType == "LSTM":
                h = h[prev_word_inds[incomplete_inds].long()]
                c = c[prev_word_inds[incomplete_inds].long()]
            elif decoderType == "GRU":
                h = h[prev_word_inds[incomplete_inds].long()]
            else:
                raise Exception("decoderType should be one of \"LSTM\", \"GRU\"!")

        encoder_out = encoder_out[prev_word_inds[incomplete_inds].long()]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            too_long = True
            print("Too long sequence!")
            break
        step += 1

    if not too_long:
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]
    else:
        try:
            seq = seqs[incomplete_inds]
            alphas = seqs_alpha[incomplete_inds]
            print("Too long sequence!")
        except:
            print("Exception while handling too long sequence!")

    return seq, alphas


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Generate Caption')

    parser.add_argument("--checkpoint_path", "-ch", type=str, help="Path to checkpoint file")
    parser.add_argument("--config_path", "-cfg", type=str, help="Path to config file")
    parser.add_argument("--word_map", "-wm", help='path to word map JSON')
    parser.add_argument("--beam_size", "-b", default=5, type=int, help='beam size for beam search')

    parser.add_argument("--img", "-i", help='path to image')
    parser.add_argument("--dont_smooth", dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()

    # Read word map
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)

    # Load word map (word2ix)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    cfgData = None
    with open(args.config_path, "r") as cfgFile:
            cfgData = yaml.safe_load(cfgFile)

    encoder, decoder = load_pretrained_model_for_inference(args.config_path, len(word_map), args.checkpoint_path)

    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, cfgData, args.img, word_map, args.beam_size)
    alphas = torch.FloatTensor(alphas)

    # Visualize caption and attention of best sequence
    visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)
