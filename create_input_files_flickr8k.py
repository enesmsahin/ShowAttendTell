from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='flickr8k',
                       karpathy_json_path='/media/enes/storage/OKUL/MMI727/PROJECT/caption_datasets/dataset_flickr8k.json',
                       image_folder='/media/enes/storage/OKUL/MMI727/PROJECT/flickr8k/originalImages/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/media/enes/storage/OKUL/MMI727/PROJECT/flickr8k/images/',
                       max_len=50)
