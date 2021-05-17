import sys
import collections
import numpy as np
import os
import errno
import zipfile
import glob

import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.utils import save_image

from lib.helpers.custom_img_folder import IndxImageFolder
from lib.dataset_metrics import DatasetMetrics


class JAFFE:
    """
    The database contains 213 images of 7 facial expressions
    (6 basic facial expressions + 1 neutral) posed by 10 Japanese female models.
    Each image has been rated on 6 emotion adjectives by 60 Japanese subjects.
    The database was planned and assembled by Michael Lyons, Miyuki Kamachi,
    and Jiro Gyoba.
    https://zenodo.org/record/3451524

    Michael J. Lyons, Shigeru Akamatsu, Miyuki Kamachi, Jiro Gyoba.
    Coding Facial Expressions with Gabor Wavelets,
    3rd IEEE International Conference on Automatic Face and Gesture Recognition,
    pp. 200-205 (1998).
    http://doi.org/10.1109/AFGR.1998.670949
    Open access content available at: https://zenodo.org/record/3430156

    Images are 256x256 gray level, in .tiff format, with no compression.
    Semantic ratings data on emotion adjectives, averaged over 60 subjects,
    are provided in the text file A_README_FIRST.txt.
    For further information on this data, please see the publication listed above.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.
    Attributes:
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
        class_to_idx (dict): Defines mapping from class names to integers.
    """

    def __init__(self, is_gpu, args):
        self.num_classes = 7
        self.gray_scale = args.gray_scale

        self.class_to_idx = {'NE': 0,
                             'HA': 1,  # happiness
                             'SA': 2,  # sadness
                             'SU': 3,  # surprise
                             'AN': 4,  # anger
                             'DI': 5,  # disgust
                             'FE': 6}  # fear
        self.idx_to_class = {0: 'NE',
                             1: 'HA',  # happiness
                             2: 'SA',  # sadness
                             3: 'SU',  # surprise
                             4: 'AN',  # anger
                             5: 'DI',  # disgust
                             6: 'FE'}  # fear

        self.__store_path = os.path.join(self.__path, 'emotion_datasets/jaffedbase.zip')
        self.__path = os.path.expanduser('datasets/Jaffe')

        self.trainset, self.valset = self.get_dataset(args.patch_size)
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

    def get_dataset(self, patch_size):
        """
        Loads and wraps training and validation datasets
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        print("Extracting JAFFE dataset")

        # download files
        try:
            os.makedirs(self.__path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        if os.path.exists(self.__store_path):
            archive = zipfile.ZipFile(self.__store_path)
            archive.extractall(self.__path)
            archive.close()

            print("Extraction successful")

            jaffe_src = os.path.join(self.__path, 'jaffedbase/')
            data = np.array(glob.glob(os.path.join(jaffe_src, "*.tiff")))

            subjects = ['KA', 'KL', 'KM', 'KR', 'MK', 'NA', 'NM', 'TM', 'UY', 'YM']
            train_subjects = subjects[:8]
            test_subjects = subjects[8:]

            train_images = []
            train_labels = []
            test_images = []
            test_labels = []

            # get train and test indices
            print("Dividing train and test")
            # create train and test folders and save audios as images
            for filepath in tqdm(data):
                # subject name, category, counter
                vp, label, _ = filepath.rstrip(".tiff").split("/")[-1].split(".")
                label = label[0:2]

                with open(filepath, 'rb') as f:
                    img = Image.open(f)
                    # scale to range (0,1)
                    img = np.asarray(img) / 255.
                    img = img.reshape(1, img.shape[0], img.shape[1])

                    if vp in train_subjects:
                        train_images.append(img)
                        train_labels.append(int(self.class_to_idx[label]))
                    elif vp in test_subjects:
                        test_images.append(img)
                        test_labels.append(int(self.class_to_idx[label]))
                    else:
                        raise Exception('Person neither in train nor in test set!')

        x_train, y_train = self.__get_jaffe(self.__path, kind='train')
        x_val, y_val = self.__get_jaffe(self.__path, kind='test')

        # up and down-sampling
        x_train = torch.nn.functional.interpolate(x_train, size=patch_size, mode='bilinear')
        x_val = torch.nn.functional.interpolate(x_val, size=patch_size, mode='bilinear')

        if not self.gray_scale:
            x_train = x_train.repeat(1, 3, 1, 1)
            x_val = x_val.repeat(1, 3, 1, 1)

        trainset = torch.utils.data.TensorDataset(x_train, y_train)
        valset = torch.utils.data.TensorDataset(x_val, y_val)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset
        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True
        Returns:
             torch.utils.data.DataLoader: train_loader, val_loader
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class CKplus:
    """
    The database contains

    1) The Images (cohn-kanade-images.zip) - there are 593 sequences across 123 subjects
    which are FACS coded at the peak frame. All sequences are from the neutral face to the
    peak expression.
    2) The Landmarks (Landmarks.zip) - All sequences are AAM tracked with 68points landmarks
    for each image.
    3) The FACS coded files (FACS_labels.zip) - for each sequence (593) there is only 1 FACS
    file, which is the last frame (the peak frame). Each line of the file corresponds to a
    specific AU and then the intensity. An example is given below.
    4) The Emotion coded files (Emotion_labels.zip) - ONLY 327 of the 593 sequences have
    emotion sequences. This is because these are the only ones the fit the prototypic
    definition. Like the FACS files, there is only 1 Emotion file for each sequence which is
    the last frame (the peak frame). There should be only one entry and the number will range
    from 0-7 (i.e. 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness,
    7=surprise). N.B there is only 327 files- IF THERE IS NO FILE IT MEANS THAT THERE IS NO
    EMOTION LABEL (sorry to be explicit but this will avoid confusion).

    P. Lucey, J.F. Cohn, T. Kanade, J. Saragih, Z. Ambadar and I. Matthews, "The Extended
    Cohn-Kanade Dataset (CK+): A complete dataset for action unit and emotion-specified
    expression", in the Proceedings of IEEE workshop on CVPR for Human Communicative Behavior
    Analysis, San Francisco, USA, 2010.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.
    Attributes:
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
        class_to_idx (dict): Defines mapping from class names to integers.
    """

    def __init__(self, is_gpu, args):
        self.num_classes = 8
        self.gray_scale = args.gray_scale
        self.args = args

        self.class_to_idx = {'NE': 0,
                             'AN': 1,  # anger
                             'CO': 2,  # contempt
                             'DI': 3,  # disgust
                             'FE': 4,  # fear
                             'HA': 5,  # happy
                             'SA': 6,  # sad
                             'SU': 7  # surprise
                             }
        self.idx_to_class = {0: 'NE',
                             1: 'AN',
                             2: 'CO',
                             3: 'DI',
                             4: 'FE',
                             5: 'HA',
                             6: 'SA',
                             7: 'SU'}

        self.__load_path = os.path.join('./datasets/CK+')
        self.__path = os.path.expanduser('./datasets/CK+')
        self.__download()

        self.trainset, self.valset = self.get_dataset(args.patch_size)
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

    def __check_exists(self):
        """
        Check if dataset has already been downloaded
        Returns:
             bool: True if downloaded dataset has been found
        """

        return os.path.exists(os.path.join(self.__path, 'train_images_tensor.pt')) and \
               os.path.exists(os.path.join(self.__path, 'train_labels_tensor.pt')) and \
               os.path.exists(os.path.join(self.__path, 'test_images_tensor.pt')) and \
               os.path.exists(os.path.join(self.__path, 'test_labels_tensor.pt'))

    def __download(self):
        """
        Downloads the CK+ dataset from the web if dataset
        hasn't already been downloaded.
        """

        if self.__check_exists():
            return

        print("Downloading CK+ dataset")

        # download files
        try:
            os.makedirs(self.__path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        if os.path.exists(os.path.join(self.__load_path, 'CK+.zip')):

            archive = zipfile.ZipFile(os.path.join(self.__load_path, 'CK+.zip'))
            archive.extractall(self.__path)
            archive.close()

            for path in ['CK+/extended-cohn-kanade-images.zip', 'CK+/Emotion_labels.zip']:
                archive = zipfile.ZipFile(os.path.join(self.__path, path))
                archive.extractall(self.__path)
                archive.close()

            print("Extraction successful")

            ck_images_src = os.path.join(self.__path, 'cohn-kanade-images')
            ck_emotion_src = os.path.join(self.__path, 'Emotion')
            emotion_data = [f for f in glob.glob(os.path.join(ck_emotion_src, '**/**/*.txt'))]

            subjects = set()
            data = []
            for e in emotion_data:
                subject = e.split('/')[-1].split('_')[0:2]
                img_path = os.path.join(subject[0], subject[1])

                subjects.add(subject[0])
                subject_img_paths = glob.glob(os.path.join(ck_images_src, os.path.join(img_path, '*.png')))
                subject_img_paths_selected = sorted(subject_img_paths)[-3:]
                neutral_path = sorted(subject_img_paths)[0]
                subject_img_paths_selected.append(neutral_path)

                f = open(e, "r")
                emotion_label = int(float(f.read().strip()))
                data.append((subject[0], emotion_label, subject_img_paths_selected))

            msk = np.random.rand(len(subjects)) < 0.8
            train_subjects = np.array(list(subjects))[msk]
            test_subjects = np.array(list(subjects))[~msk]

            train_images = []
            train_labels = []
            test_images = []
            test_labels = []

            # get train and test indices
            print("Dividing train and test")
            # create train and test folders and save images
            for s, label, img_paths in tqdm(data):

                imgs = []
                for filepath in img_paths:
                    with open(filepath, 'rb') as f:
                        # 490x640, 480x640, 480x720
                        img = Image.open(f).resize((self.args.patch_size, self.args.patch_size))  # (640, 480))
                        if img.mode == 'RGB':
                            img = img.convert('RGB')

                        # scale to range (0,1)
                        img = np.asarray(img) / 255.

                        if len(img.shape) == 2:
                            img = img.reshape(1, img.shape[0], img.shape[1])
                            img = img.repeat(3, axis=0)
                        else:
                            img = img.transpose(2, 0, 1)
                        imgs.append(img)

                if s in train_subjects:
                    for ind, img in enumerate(imgs):
                        train_images.append(img)
                        if ind < len(imgs) - 1:
                            train_labels.append(label)
                        else:
                            # neutral is the first image of the sequence,
                            # appended at last position
                            train_labels.append(0)

                elif s in test_subjects:
                    for ind, img in enumerate(imgs):
                        test_images.append(img)
                        if ind < len(imgs) - 1:
                            test_labels.append(label)
                        else:
                            test_labels.append(0)
                else:
                    raise Exception('Person neither in train nor in test set!')

            train_images = torch.Tensor(train_images).float()
            train_labels = torch.Tensor(train_labels).long()
            test_images = torch.Tensor(test_images).float()
            test_labels = torch.Tensor(test_labels).long()

            torch.save(train_images, os.path.join(self.__path, 'train_images_tensor.pt'))
            torch.save(train_labels, os.path.join(self.__path, 'train_labels_tensor.pt'))
            torch.save(test_images, os.path.join(self.__path, 'test_images_tensor.pt'))
            torch.save(test_labels, os.path.join(self.__path, 'test_labels_tensor.pt'))

            print('Done!')
        else:
            print("Please download the dataset to datasets/CK+")

    def __get_ckplus(self, path, kind='train'):
        """
        Load CK+ data
        Parameters:
            path (str): Base directory path containing .npy files for
                the CK+ dataset
            kind (str): Accepted types are 'train' and 'validation' for
                training and validation set stored in .npy files
        Returns:
            numpy.array: images, labels
        """

        images = torch.load(os.path.join(path, kind + '_images_tensor.pt'))
        labels = torch.load(os.path.join(path, kind + '_labels_tensor.pt'))

        return images, labels

    def get_dataset(self, patch_size):
        """
        Loads and wraps training and validation datasets
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        x_train, y_train = self.__get_ckplus(self.__path, kind='train')
        x_val, y_val = self.__get_ckplus(self.__path, kind='test')

        # up and down-sampling
        x_train = torch.nn.functional.interpolate(x_train, size=patch_size, mode='bilinear')
        x_val = torch.nn.functional.interpolate(x_val, size=patch_size, mode='bilinear')

        trainset = torch.utils.data.TensorDataset(x_train, y_train)
        valset = torch.utils.data.TensorDataset(x_val, y_val)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset
        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True
        Returns:
             torch.utils.data.DataLoader: train_loader, val_loader
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class FER2013:
    """
    The database is based on the kaggle challenge on emotion recognition:
    https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

    The data consists of 48x48 pixel grayscale images of faces.
    The faces have been automatically registered so that the face is more or less
    centered and occupies about the same amount of space in each image.
    The task is to categorize each face based on the emotion shown in the
    facial expression in to one of seven categories
    (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

    For improved labels see: https://github.com/microsoft/FERPlus


    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.
    Attributes:
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
        class_to_idx (dict): Defines mapping from class names to integers.
    """

    def __init__(self, is_gpu, args):
        self.num_classes = 7
        self.gray_scale = args.gray_scale

        self.class_to_idx = {'AN': 0,  # anger
                             'DI': 1,  # disgust
                             'FE': 2,  # fear
                             'HA': 3,  # happy
                             'SA': 4,  # sad
                             'SU': 5,  # surprise
                             'NE': 6  # neutral
                             }
        self.idx_to_class = {0: 'AN',
                             1: 'DI',
                             2: 'FE',
                             3: 'HA',
                             4: 'SA',
                             5: 'SU',
                             6: 'NE'}

        self.__load_path = os.path.join('./datasets/FER2013')
        self.__path = os.path.expanduser('./datasets/FER2013')
        self.__download()

        self.trainset, self.valset = self.get_dataset(args.patch_size)
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

    def __check_exists(self):
        """
        Check if dataset has already been downloaded
        Returns:
             bool: True if downloaded dataset has been found
        """

        return os.path.exists(os.path.join(self.__path, 'train_images_tensor.pt')) and \
               os.path.exists(os.path.join(self.__path, 'train_labels_tensor.pt')) and \
               os.path.exists(os.path.join(self.__path, 'test_images_tensor.pt')) and \
               os.path.exists(os.path.join(self.__path, 'test_labels_tensor.pt'))

    def __download(self):
        """
        Downloads the FER2013 dataset from the web if dataset
        hasn't already been downloaded.
        """

        if self.__check_exists():
            return

        print("Downloading FER2013 dataset")

        # download files
        try:
            os.makedirs(self.__path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # https://www.kaggle.com/deadskull7/fer2013/notebooks
        if os.path.exists(os.path.join(self.__load_path, 'FER2013-kaggle.zip')):

            archive = zipfile.ZipFile(os.path.join(self.__load_path, 'FER2013-kaggle.zip'))
            archive.extractall(self.__path)
            archive.close()

            print("Extraction successful")

            # https://www.kaggle.com/egreblova/facial-expression-recognition-with-resnet-models
            all_data = pd.read_csv("./datasets/FER2013/fer2013.csv")

            # split to 3 parts
            groups = [g for _, g in all_data.groupby('Usage')]
            training_data = groups[2]
            validation_data = groups[1]
            testing_data = groups[0]

            data_images = []
            data_labels = []

            for data in [training_data, validation_data]:
                images, labels = data['pixels'], data['emotion']
                # normalizing data to be between 0 and 1
                images = np.array([np.fromstring(image, np.uint8, sep=' ') for image in images]) / 255.0
                # 1 color channel, 48x48 images
                images = images.reshape(images.shape[0], 1, 48, 48)
                labels = np.array(labels)

                data_images.append(images)
                data_labels.append(labels)

            train_images = torch.Tensor(data_images[0]).float()
            train_labels = torch.Tensor(data_labels[0]).long()
            test_images = torch.Tensor(data_images[1]).float()
            test_labels = torch.Tensor(data_labels[1]).long()

            torch.save(train_images, os.path.join(self.__path, 'train_images_tensor.pt'))
            torch.save(train_labels, os.path.join(self.__path, 'train_labels_tensor.pt'))
            torch.save(test_images, os.path.join(self.__path, 'test_images_tensor.pt'))
            torch.save(test_labels, os.path.join(self.__path, 'test_labels_tensor.pt'))

            print('Done!')
        else:
            print("Please download the dataset to datasets/CK+")

    def __get_fer2013(self, path, kind='train'):
        """
        Load FER2013 data
        Parameters:
            path (str): Base directory path containing .npy files for
                the CK+ dataset
            kind (str): Accepted types are 'train' and 'validation' for
                training and validation set stored in .npy files
        Returns:
            numpy.array: images, labels
        """

        images = torch.load(os.path.join(path, kind + '_images_tensor.pt'))
        labels = torch.load(os.path.join(path, kind + '_labels_tensor.pt'))

        return images, labels

    def get_dataset(self, patch_size):
        """
        Loads and wraps training and validation datasets
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        x_train, y_train = self.__get_fer2013(self.__path, kind='train')
        x_val, y_val = self.__get_fer2013(self.__path, kind='test')

        # up and down-sampling
        x_train = torch.nn.functional.interpolate(x_train, size=patch_size, mode='bilinear')
        x_val = torch.nn.functional.interpolate(x_val, size=patch_size, mode='bilinear')

        if not self.gray_scale:
            x_train = x_train.repeat(1, 3, 1, 1)
            x_val = x_val.repeat(1, 3, 1, 1)

        trainset = torch.utils.data.TensorDataset(x_train, y_train)
        valset = torch.utils.data.TensorDataset(x_val, y_val)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset
        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True
        Returns:
             torch.utils.data.DataLoader: train_loader, val_loader
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader


class RAFbasic:
    """
    Real-world Affective Faces Database (RAF-DB) is a
    large-scale facial expression database with around 30K
    great-diverse facial images downloaded from the Internet.
    http://www.whdeng.cn/RAF/model1.html

    The data consists of
    - 29672 number of real-world images,
    - a 7-dimensional expression distribution vector for each image,
    - two different subsets: single-label subset, including 7 classes of
    basic emotions; two-tab subset, including 12 classes of compound emotions,
    - 5 accurate landmark locations, 37 automatic landmark locations, bounding box,
    race, age range and gender attributes annotations per image,
    - baseline classifier outputs for basic emotions and compound emotions.


    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.
    Attributes:
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
        class_to_idx (dict): Defines mapping from class names to integers.
    """

    def __init__(self, is_gpu, args):
        self.num_classes = 7
        self.gray_scale = args.gray_scale

        self.class_to_idx = {'SU': 0,  # surprise
                             'FE': 1,  # fear
                             'DI': 2,  # disgust
                             'HA': 3,  # happiness
                             'SA': 4,  # sadness
                             'AN': 5,  # anger
                             'NE': 6  # neutral
                             }
        self.idx_to_class = {0: 'SU',  # surprise
                             1: 'FE',  # fear
                             2: 'DI',  # disgust
                             3: 'HA',  # happiness
                             4: 'SA',  # sadness
                             5: 'AN',  # anger
                             6: 'NE'  # neutral
                             }

        self.indices2img_path = []
        self.indices2labels = []

        self.__load_path = os.path.join('./datasets/RAF')
        self.__path = os.path.expanduser('./datasets/RAF')
        self.__download()

        self.trainset, self.valset = self.get_dataset(args.patch_size)
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

    def __check_exists(self):
        """
        Check if dataset has already been downloaded
        Returns:
             bool: True if downloaded dataset has been found
        """

        return os.path.exists(os.path.join(self.__path, 'train_images_tensor.pt')) and \
               os.path.exists(os.path.join(self.__path, 'train_labels_tensor.pt')) and \
               os.path.exists(os.path.join(self.__path, 'test_images_tensor.pt')) and \
               os.path.exists(os.path.join(self.__path, 'test_labels_tensor.pt'))

    def __download(self):
        """
        Downloads the FER2013 dataset from the web if dataset
        hasn't already been downloaded.
        """

        if self.__check_exists():
            return

        print("Downloading RAFbasic dataset")

        # download files
        try:
            os.makedirs(self.__path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        if os.path.exists(os.path.join(self.__load_path, 'raf_basic.zip')):

            archive = zipfile.ZipFile(os.path.join(self.__load_path, 'raf_basic.zip'))
            archive.extractall(self.__path)
            archive.close()

            img_path = os.path.join(self.__path, 'basic/Image/aligned')
            archive = zipfile.ZipFile(os.path.join(self.__path, 'basic/Image/aligned.zip'))
            archive.extractall(os.path.join(self.__path, 'basic/Image'))
            archive.close()

            print("Extraction successful")

            images = np.array(glob.glob(os.path.join(img_path, "*.jpg")))

            label_path = os.path.join(self.__path, 'basic/EmoLabel/list_patition_label.txt')

            with open(label_path) as f:
                data_read = f.read()
            f.close()

            label_strings = data_read.splitlines()
            labels = {}

            for lab in label_strings:
                pic_name, label = lab.split()
                labels[pic_name] = label

            train_images = []
            train_labels = []
            train_indices = []
            test_images = []
            test_labels = []
            test_indices = []

            # get train and test indices
            print("Dividing train and test")
            # create train and test folders and save audios as images
            for img_indx, filepath in enumerate(tqdm(images)):
                # subject name, category, counter
                pure_img_name = filepath.rstrip(".jpg").rstrip("_aligned").split('/')[-1] + ".jpg"
                data_type = pure_img_name.split("_")[0]
                label = labels[pure_img_name]
                self.indices2img_path.append(filepath)
                self.indices2labels.append(int(label) - 1)

                with open(filepath, 'rb') as f:
                    img = Image.open(f)
                    # scale to range (0,1)
                    if img.mode == 'RGB':
                        img = img.convert('RGB')

                    # scale to range (0,1)
                    img = np.asarray(img) / 255.

                    if len(img.shape) == 2:
                        img = img.reshape(1, img.shape[0], img.shape[1])
                        img = img.repeat(3, axis=0)
                    else:
                        img = img.transpose(2, 0, 1)

                    if data_type == "train":
                        train_images.append(img)
                        # since labels are 1-7, but we want 0-6, we subtract 1
                        train_labels.append(int(label) - 1)
                        train_indices.append(img_indx)
                    elif data_type == "test":
                        test_images.append(img)
                        test_labels.append(int(label) - 1)
                        test_indices.append(img_indx)
                    else:
                        raise Exception('Person neither in train nor in test set!')

            train_images = torch.Tensor(train_images).float()
            train_labels = torch.Tensor(train_labels).long()
            train_indices = torch.Tensor(train_indices).long()
            test_images = torch.Tensor(test_images).float()
            test_labels = torch.Tensor(test_labels).long()
            test_indices = torch.Tensor(test_indices).long()

            torch.save(train_images, os.path.join(self.__path, 'train_images_tensor.pt'))
            torch.save(train_labels, os.path.join(self.__path, 'train_labels_tensor.pt'))
            torch.save(train_indices, os.path.join(self.__path, 'train_indices_tensor.pt'))

            torch.save(test_images, os.path.join(self.__path, 'test_images_tensor.pt'))
            torch.save(test_labels, os.path.join(self.__path, 'test_labels_tensor.pt'))
            torch.save(test_indices, os.path.join(self.__path, 'test_indices_tensor.pt'))

            with open(os.path.join(self.__path, "indices2img_path.json"), 'w') as f:
                # indent=2 is not needed but makes the file human-readable
                json.dump(self.indices2img_path, f, indent=2)

            with open(os.path.join(self.__path, "indices2labels.json"), 'w') as f:
                # indent=2 is not needed but makes the file human-readable
                json.dump(self.indices2labels, f, indent=2)

            print('Done!')
        else:
            print("Please download the dataset to datasets/RAF")

    def __get_raf_basic(self, path, kind='train'):
        """
        Load RAFbasic data
        Parameters:
            path (str): Base directory path containing .npy files for
                the RAF dataset
            kind (str): Accepted types are 'train' and 'validation' for
                training and validation set stored in .npy files
        Returns:
            numpy.array: images, labels
        """

        if len(self.indices2labels) < 1:  # there are no elements
            with open(os.path.join(self.__path, "indices2img_path.json"), 'r') as f:
                self.indices2img_path = json.load(f)
            with open(os.path.join(self.__path, "indices2labels.json"), 'r') as f:
                self.indices2labels = json.load(f)

        images = torch.load(os.path.join(path, kind + '_images_tensor.pt'))
        labels = torch.load(os.path.join(path, kind + '_labels_tensor.pt'))
        indices = torch.load(os.path.join(path, kind + '_indices_tensor.pt'))

        return images, labels, indices

    def get_dataset(self, patch_size):
        """
        Loads and wraps training and validation datasets
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        x_train, y_train, indices_train = self.__get_raf_basic(self.__path, kind='train')
        x_val, y_val, indices_test = self.__get_raf_basic(self.__path, kind='test')

        # up and down-sampling
        x_train = torch.nn.functional.interpolate(x_train, size=patch_size, mode='bilinear')
        x_val = torch.nn.functional.interpolate(x_val, size=patch_size, mode='bilinear')

        trainset = torch.utils.data.TensorDataset(x_train, y_train, indices_train)
        valset = torch.utils.data.TensorDataset(x_val, y_val, indices_test)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset
        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True
        Returns:
             torch.utils.data.DataLoader: train_loader, val_loader
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader