import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
from models import Encoder, EncoderWide, EncoderFPN, Decoder, EncoderFPN2, Decoder2layer
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import yaml
import os

out_dir = "/home/deepuser/deepnas/DISK4/DISK4/enes/mmi727_project/trainings/7/"
config_path = out_dir + "config.yaml"

log_path = os.path.join(out_dir, "./training_results")
if not os.path.exists(log_path):
    os.makedirs(log_path)

summaryWriter = SummaryWriter(log_path)

# Data parameters
img_data_folder = '/home/deepuser/deepnas/DISK4/DISK4/enes/mmi727_project/coco/images'  # folder with data files saved by create_input_files.py
img_data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

cfgData = None
with open(config_path, "r") as cfgFile:
        cfgData = yaml.safe_load(cfgFile)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

print_freq = 50  # print training/validation stats every __ batches
start_epoch = 0 # TODO remove this, it is unnecessary
epochs_since_improvement = 0 # keeps track of number of epochs since there's been an improvement in validation BLEU
best_bleu4 = 0. # Best BLEU-4 score until now

modelTypes = cfgData["Model Type"]
modelParams = cfgData["Model Parameters"]
trainParams = cfgData["Training Parameters"]

def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, start_epoch, img_data_name, word_map

    # Read word map
    word_map_file = os.path.join(img_data_folder, 'WORDMAP_' + img_data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # ***TODO: set fine_tune for necessary encoders
    # ***TODO: remove fpn2 in train.py, models.py, eval.py
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
                            vocab_size=len(word_map),
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
                                    vocab_size=len(word_map),
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
    
    # Initialize / load checkpoint
    if trainParams["checkpoint"] is None:
        decoder_optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=trainParams["decoder_learning_rate"])
        encoder_optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=trainParams["encoder_learning_rate"]) if trainParams["fine_tune_encoder"] else None

    else:
        checkpoint = torch.load(trainParams["checkpoint"])
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        encoder_state_dict = checkpoint['encoder_state_dict']
        encoder_optimizer_state_dict = checkpoint['encoder_optimizer_state_dict']
        decoder_state_dict = checkpoint['decoder_state_dict']
        decoder_optimizer_state_dict = checkpoint['decoder_optimizer_state_dict']
        
        encoder.load_state_dict(encoder_state_dict)
        decoder.load_state_dict(decoder_state_dict)

        # Move to GPU, if available
        decoder = decoder.to(device)
        encoder = encoder.to(device)

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

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(img_data_folder, img_data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=trainParams["batch_size"], shuffle=True, num_workers=1, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(img_data_folder, img_data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=trainParams["batch_size"], shuffle=True, num_workers=1, pin_memory=True)

    # Epochs
    for epoch in range(trainParams["start_epoch"], trainParams["num_epochs"]):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        # if epochs_since_improvement == 20:
        #     break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if trainParams["fine_tune_encoder"]:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion,
                                epoch=epoch)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(img_data_name, epoch, epochs_since_improvement, encoderType, decoderType, enable2LayerDecoder, attentionType, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best, epoch, out_dir)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        # ***too many values to unpack error https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/issues/86
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += trainParams["alpha_c"] * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if trainParams["grad_clip"] is not None:
            clip_gradient(decoder_optimizer, trainParams["grad_clip"])
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, trainParams["grad_clip"])

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))

            summaryWriter.add_scalar("train_loss_curr", losses.val, epoch * len(train_loader) + i)
            summaryWriter.add_scalar("train_loss_avg", losses.avg, epoch * len(train_loader) + i)
            summaryWriter.add_scalar("train_top5_acc_curr", top5accs.val, epoch * len(train_loader) + i)
            summaryWriter.add_scalar("train_top5_acc_avg", top5accs.avg, epoch * len(train_loader) + i)


def validate(val_loader, encoder, decoder, criterion, epoch):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param epoch: epoch number
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            # ***too many values to unpack error https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/issues/86
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += trainParams["alpha_c"] * ((1. - alphas.sum(dim=1)) ** 2).mean()
            
            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

        summaryWriter.add_scalar("val_loss_avg", losses.avg, epoch)
        summaryWriter.add_scalar("val_top5_acc_avg", top5accs.avg, epoch)
        summaryWriter.add_scalar("val_bleu4", bleu4, epoch)

    return bleu4


if __name__ == '__main__':
    main()
