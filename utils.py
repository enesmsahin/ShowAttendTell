import os
import numpy as np
import h5py
import json
import torch
from imageio import imread
from PIL import Image
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
from shutil import copy

from models import Encoder, EncoderWide, EncoderFPN, Decoder, Decoder2layer
import yaml

def create_model_for_training(config_path, vocab_size):
    """!
    Creates a model and returns encoder, decoder

    @param config_path: config.yaml for model to be loaded
    @param vocab_size: size of vocabulary
    @return encoder, decoder models. Loaded to the GPU if available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

    # Load model
    cfgData = None
    with open(config_path, "r") as cfgFile:
            cfgData = yaml.safe_load(cfgFile)

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
    else:
        raise Exception("Encoder Type must be one of \"default\", \"wide\", \"fpn\".")

    encoder.fine_tune(trainParams["fine_tune_encoder"])
        
    decoderType = modelTypes["Decoder"]
    enable2LayerDecoder = modelTypes["Enable2LayerDecoder"]
    attentionType = modelTypes["Attention"]

    encoder_dim = 2048

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

    
    # load model and create necessary optimizers for training
    # Initialize / load checkpoint for training
    epochs_since_improvement = 0
    best_bleu4 = 0
    if trainParams["checkpoint"] is None: # if pretrained model is not available
        decoder_optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                            lr=trainParams["decoder_learning_rate"])
        encoder_optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                            lr=trainParams["encoder_learning_rate"]) if trainParams["fine_tune_encoder"] else None

    else: # if pretrained model is available
        checkpoint = torch.load(trainParams["checkpoint"])
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        encoder_state_dict = checkpoint['encoder_state_dict']
        encoder_optimizer_state_dict = checkpoint['encoder_optimizer_state_dict']
        decoder_state_dict = checkpoint['decoder_state_dict']
        decoder_optimizer_state_dict = checkpoint['decoder_optimizer_state_dict']
        
        encoder.load_state_dict(encoder_state_dict)
        decoder.load_state_dict(decoder_state_dict)

        decoder_optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, decoder.parameters()))
        decoder_optimizer.load_state_dict(decoder_optimizer_state_dict)

        encoder_optimizer = None
        if trainParams["fine_tune_encoder"] is True:
            if encoder_optimizer_state_dict is None:
                encoder.fine_tune(True)
                encoder_optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                lr=trainParams["encoder_learning_rate"])
            else:
                encoder_optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                lr=trainParams["encoder_learning_rate"])
                encoder_optimizer.load_state_dict(encoder_optimizer_state_dict)

    encoder.train()
    decoder.train()

    # Move to GPU, if available
    encoder = encoder.to(device)
    decoder = decoder.to(device)        

    return encoder, decoder, encoder_optimizer, decoder_optimizer, epochs_since_improvement, best_bleu4, encoderType, decoderType, attentionType, enable2LayerDecoder

def load_pretrained_model_for_inference(config_path, vocab_size, checkpoint_path=None):
    """!
    Loads a pretrained model and returns encoder, decoder

    @param config_path: config.yaml for model to be loaded
    @param vocab_size: size of vocabulary
    @param checkpoint_path: path to pretrained model. If None, weights are initialized from scratch
    @return encoder, decoder models. Loaded to the GPU if available.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

    # Load model
    cfgData = None
    with open(config_path, "r") as cfgFile:
            cfgData = yaml.safe_load(cfgFile)

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
    else:
        raise Exception("Encoder Type must be one of \"default\", \"wide\", \"fpn\".")

    encoder.fine_tune(trainParams["fine_tune_encoder"])
        
    decoderType = modelTypes["Decoder"]
    enable2LayerDecoder = modelTypes["Enable2LayerDecoder"]
    attentionType = modelTypes["Attention"]

    encoder_dim = 2048

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

    if checkpoint_path != None:
        checkpoint = torch.load(checkpoint_path)
        encoder_state_dict = checkpoint['encoder_state_dict']
        decoder_state_dict = checkpoint['decoder_state_dict']
        encoder.load_state_dict(encoder_state_dict)
        decoder.load_state_dict(decoder_state_dict)

    encoder.eval()
    decoder.eval()

    # Move to GPU, if available
    encoder = encoder.to(device)
    decoder = decoder.to(device)        

    return encoder, decoder 

def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            # ***Copy images
            # if split == 'VAL' or split == 'TEST':
            #     outImgPath = os.path.join(output_folder, split.lower())
            #     if not os.path.exists(outImgPath):
            #         os.makedirs(outImgPath)

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                # img = imresize(img, (256, 256))
                img = np.array(Image.fromarray(img).resize((256,256), resample=Image.BILINEAR))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # ***Copy images
                # if split == 'VAL' or split == 'TEST':
                #     copy(impaths[i], outImgPath)

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoderType, decoderType, enable2LayerDecoder, attentionType, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best, id, outDir):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    :param id: checkpoint id
    :out_path: path to output directory
    """

    """
    # *** error might be encountered https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/issues/86#issuecomment-622783546
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    """

    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoderType': encoderType,
             'decoderType': decoderType,
             'enable2LayerDecoder': enable2LayerDecoder,
             'attentionType': attentionType,
             'encoder_state_dict': encoder.state_dict(),
             'decoder_state_dict': decoder.state_dict(),
             'encoder_optimizer_state_dict': encoder_optimizer.state_dict() if encoder_optimizer is not None else None,
             'decoder_optimizer_state_dict': decoder_optimizer.state_dict() if decoder_optimizer is not None else None}

    prefix = "BEST_" if is_best else ""
    filename = prefix + 'checkpoint_' + str(id) + '_' + data_name + '.pth.tar'
    outFile = os.path.join(outDir, filename)
    torch.save(state, outFile)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
