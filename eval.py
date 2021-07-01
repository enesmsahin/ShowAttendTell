import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
from models import Encoder, EncoderWide, EncoderFPN, Decoder, EncoderFPN2, Decoder2layer
import os
import yaml
import argparse

parser = argparse.ArgumentParser("Evaluation script!")

def evaluate(beam_size, checkpoint_path, config_path):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # model_path = "/home/deepuser/deepnas/DISK4/DISK4/enes/mmi727_project/trainings/1/"

    # checkpoint_path = os.path.join(model_path, "BEST_checkpoint_10_coco_5_cap_per_img_5_min_word_freq.pth.tar") # model checkpoint
    # config_path = os.path.join(model_path, "config.yaml")

    # Data parameters
    img_data_folder = '/home/deepuser/deepnas/DISK4/DISK4/enes/mmi727_project/coco/images'  # folder with data files saved by create_input_files.py
    img_data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

    word_map_file = os.path.join(img_data_folder, 'WORDMAP_' + img_data_name + '.json')  # word map, ensure it's the same the data was encoded with and the model was trained with

    cfgData = None
    with open(config_path, "r") as cfgFile:
            cfgData = yaml.safe_load(cfgFile)

    # Load word map (word2ix)
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

    modelTypes = cfgData["Model Type"]
    modelParams = cfgData["Model Parameters"]
    trainParams = cfgData["Training Parameters"]


    encoder = None
    encoderType = modelTypes["Encoder"]
    endodedImageSize = modelParams["encoded_image_size"]
    if encoderType == "default":
        encoder = Encoder(endodedImageSize)
    elif encoderType == "wide":
        encoder = EncoderWide(endodedImageSize)
    elif encoderType == "fpn":
        encoder = EncoderFPN(endodedImageSize)
    elif encoderType == "fpn2":
        encoder = EncoderFPN2(endodedImageSize)
    else:
        raise Exception("Encoder Type must be one of \"default\", \"wide\", \"fpn\".")

    encoder.fine_tune(trainParams["fine_tune_encoder"])
        
    decoderType = modelTypes["Decoder"]
    enable2LayerDecoder = modelTypes["Enable2LayerDecoder"]
    attentionType = modelTypes["Attention"]

    encoder_dim = 2048 if encoderType != "fpn2" else 1024

    if not enable2LayerDecoder:
        decoder = Decoder(  
                            attention_dim=modelParams["attention_dim"],
                            embed_dim=modelParams["embedding_dim"],
                            decoder_dim=modelParams["decoder_dim"],
                            vocab_size=vocab_size,
                            dropout=modelParams["dropout"],
                            encoder_dim=encoder_dim,
                            decoderType=decoderType,
                            attentionType=attentionType
                        )
    else:
        decoder = Decoder2layer(  
                                    attention_dim=modelParams["attention_dim"],
                                    embed_dim=modelParams["embedding_dim"],
                                    decoder_dim=modelParams["decoder_dim"],
                                    vocab_size=vocab_size,
                                    dropout=modelParams["dropout"],
                                    encoder_dim=encoder_dim,
                                    decoderType=decoderType,
                                    attentionType=attentionType
                                )
    print("Encoder Type: " + encoderType)
    print("Decoder Type: " + decoderType)
    print("Attention Type: " + attentionType)
    print("Encoder Dim: " + str(encoder_dim))
    print("Enable2LayerDecoder: " + str(enable2LayerDecoder))

    checkpoint = torch.load(checkpoint_path)
    encoder_state_dict = checkpoint['encoder_state_dict']
    decoder_state_dict = checkpoint['decoder_state_dict']
    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)

    # Move to GPU, if available
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()

    # Normalization transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])


    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(img_data_folder, img_data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
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

        # Lists to store completed sequences and scores
        complete_seqs = list()
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
                awe, _ = decoder.attention(encoder_out, h_2)  # (s, encoder_dim), (s, num_pixels) # attention weighted encoding
                gate = decoder.sigmoid(decoder.f_beta(h_2))  # gating scalar, (s, encoder_dim)
            else:
                awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels) # attention weighted encoding
                gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)

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

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]

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
        else:
            try:
                seq = seqs[incomplete_inds]
                print("Too long sequence!")
            except:
                print("Exception while handling too long sequence!")
                print("Moving onto next image!")
                continue

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    return bleu4


if __name__ == '__main__':
    parser.add_argument("-b", "--beam_size", type=int, help="Beam size to be used in beam search.")
    parser.add_argument("-ch", "--checkpoint_path", type=str, help="Path to checkpoint file")
    parser.add_argument("-cfg", "--config_path", type=str, help="Path to config file")
    args = parser.parse_args()

    beam_size = args.beam_size
    checkpoint_path = args.checkpoint_path
    config_path = args.config_path

    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size, checkpoint_path, config_path)))
