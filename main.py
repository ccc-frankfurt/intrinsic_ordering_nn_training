import lib.architectures as architectures
import lib.datasets as datasets

from lib.helpers.initialization import WeightInit
from lib.helpers.labels_preprocessing import labels2one_hot
from lib.dataset_metrics import *
from lib.trainer import *
from lib.training_metrics import *
from lib.visualization import *

import datetime
import pickle
from cmdparser import parser
import matplotlib.backends.backend_pdf

import torch.optim as optim


args = parser.parse_args()

if args.save_dir != '':
    save_dir = args.save_dir
else:
    save_dir = '_'.join(str(datetime.datetime.now()).split(' '))

script_dir = Path(__file__).parent
save_path = os.path.join(os.path.join(str(script_dir), os.path.join(os.path.join('results', args.dataset)),
                         args.architecture), save_dir)

# create log file
# if just visualization - no need to create a file, else it will overwrite the training info
if args.train_networks:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    log_file = os.path.join(save_path, "stdout.txt")

    # write parsed args to log file
    log = open(log_file, "a")
    for arg in vars(args):
        print(arg, getattr(args, arg))
        log.write(arg + ':' + str(getattr(args, arg)) + '\n')
    log.close()

dataset = None
is_gpu = torch.cuda.is_available()
# 1. load dataset
print("Load/create dataset with indices")
dataset_init_method = getattr(datasets, args.dataset)
dataset = dataset_init_method(is_gpu, args)
# Get a sample input from the data loader to infer color channels/size
net_input, net_classes, _ = next(iter(dataset.train_loader))
# get the amount of color channels in the input images
args.num_colors = net_input.size(1)
multilabel = True if len(net_classes.shape) > 1 else False

# 2. visualize data
print("Visualize data")
vis = Visualizer()
# when computing dataset metrics, for datasets with uneven img sizes the batch-size=1
# so there will be too many files and a "too many files open" error might occur
if not args.compute_dataset_metrics and not args.batch_size == 1:
    vis.check_images(dataset)
    '''if not args.multilabel:
        vis.check_dataset_class_balance(dataset)'''

# 3. calculate dataset metrics, like img entropy, segment count etc.
print("Calculate dataset metrics")

m = DatasetMetrics(dataset, args.dataset)
if args.compute_dataset_metrics:
    m.segment_count()
    m.save()
    m.img_entropy()
    m.save()
    m.img_frequency(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    m.save()
    if args.dataset == "CIFAR10":
        m.human_uncertainty_CIFAR10()
    if args.dataset == "VOCDetection":
        m.additional_metrics_VOCDetection()
    m.save()

    m.edge_strength()
    m.save()
print(m.evaluation)


metrics_train = m.evaluation[m.evaluation.index.str.contains('/train/', regex=False)]
metrics_test = m.evaluation[m.evaluation.index.str.contains('/test/', regex=False)]

# plot metric histograms
if 'entropy' in m.evaluation:
    vis.visualize_metric_histogram(metrics_train['entropy'], 'Entropy', 'Bits', 50,
                               os.path.join(dataset.name,'train'))
if 'edge_strength' in m.evaluation:
    vis.visualize_metric_histogram(metrics_train['edge_strength'], 'Edge strengths',
                               'Summed edge strengths', 50,
                               os.path.join(dataset.name,'train'))
if 'freq_coeff_percentage' in m.evaluation:
    vis.visualize_metric_histogram(metrics_train['freq_coeff_percentage'], 'Frequency',
                               'Frequency % coeff needed', 50,
                               os.path.join(dataset.name,'train'))
if args.dataset == 'ImageNet':
    if 'segcount' in m.evaluation:
        vis.visualize_metric_histogram(metrics_train['segcount'], 'Segment count', 'Segment count', 20,
                                   os.path.join(dataset.name,'train'))

if args.dataset == 'KTH_TIPS':
    if 'segcount' in m.evaluation:
        vis.visualize_metric_histogram(metrics_train['segcount'], 'Segment count', 'Segment count', 10,
                                   os.path.join(dataset.name,'train'))

if args.dataset == 'CIFAR10':
    if 'segcount' in m.evaluation:
        vis.visualize_metric_histogram(metrics_train['segcount'], 'Segment count', 'Segment count', 10,
                                   os.path.join(dataset.name,'train'))
    if 'human_uncertainty' in m.evaluation:
        vis.visualize_metric_histogram(metrics_test['human_uncertainty'],
                                   'Pred. entropy of human uncertainty',
                                   'Bits', 15,
                                   os.path.join(dataset.name,'test'))

if args.dataset == 'VOCDetection':
    if 'segcount' in m.evaluation:
        vis.visualize_metric_histogram(metrics_train['segcount'], 'Segment count', 'Segment count', 50,
                                   os.path.join(dataset.name,'train'))
    if 'img_difficulty' in m.evaluation:
        vis.visualize_metric_histogram(metrics_train['img_difficulty'][m.evaluation['img_difficulty'].notnull()],
                                   'Image difficulty',
                                   'Human response time in seconds', 50,
                                   os.path.join(dataset.name,'train'))
    if 'obj_sizes' in m.evaluation:
        vis.visualize_metric_histogram(metrics_train['obj_sizes'],
                                   'Object sizes',
                                   'Bounding box area', 15,
                                   os.path.join(dataset.name,'train'))
    if 'num_instances' in m.evaluation:
        vis.visualize_metric_histogram(metrics_train['num_instances'],
                                   'Number of instances',
                                   'Number of instances', [1,2,3,4,8],
                                   os.path.join(dataset.name,'train'))

# 4a. NN training

if args.train_networks:
    print("NN training")

    # if random labels - store them in order to compare later
    if args.randomize_labels and not os.path.exists(os.path.join(save_path, 'random_labels.csv')):
        random_labels = pd.DataFrame.from_dict({'img_paths':list(dataset.random_labels.keys()),
                                                'labels':list(dataset.random_labels.values())})
        random_labels.set_index('img_paths', inplace=True)
        random_labels.to_csv(os.path.join(save_path, 'random_labels.csv'))

    if args.resume:
        checkpoint = torch.load(os.path.join(save_path, 'model_parameters.tar'))
        start_network = int(checkpoint['net_name'].split(' ')[-1])
    else:
        start_network = 0

    # compute for every network the agreement of correct indices
    # train several networks. in this case - same architecture/different initialization
    for i in range(start_network, args.num_networks, 1):

        # import model from architectures class
        if args.architecture == "DenseNet":
            net = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=False)
            net.classifier = nn.Linear(1024, dataset.num_classes)
        else:
            net_init_method = getattr(architectures, args.architecture)
            net = net_init_method(num_classes=len(dataset.idx_to_class), num_channels=args.num_colors, args=args)
        print(net)

        # Initialize the weights of the model, by default according to He et al.
        print("Initializing network with: " + args.weight_init)
        WeightInitializer = WeightInit(args.weight_init)
        WeightInitializer.init_model(net)

        net_name = args.architecture + ' ' + str(i)
        print("Training architecture "+ str(net_name))

        if multilabel:
            # combines a Sigmoid layer and the BCELoss in one single class
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        # optimizer and scheduler for VOCDetection
        # https://arxiv.org/pdf/2009.14119v2.pdf
        # https://openreview.net/pdf?id=KsN9p5qJN3
        if args.optimizer_type == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
        elif args.optimizer_type == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=args.learning_rate,
                              momentum=args.sgd_momentum, weight_decay=args.weight_decay)

        '''
        # optimizer and scheduler used for CIFAR10
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)'''

        t = Trainer(dataset, net, optimizer, criterion, net_name, scheduler_type=args.scheduler_type,
                    save_path=save_path, resume=args.resume and (start_network == i),
                    metrics=m.evaluation,
                    args=args)
        # after each epoch, test is also called in train
        # and correct_indices are saved for both train and test
        t.train(args.epochs)

        if not args.multilabel:
            t.check_accuracy()

# 4b
def load_indices_correct(save_path, data_mode, indices_type='network_indices_correct'):
    networks_indices_correct = []

    for file in os.listdir(save_path):
        if file.endswith(".pkl") and indices_type in file and data_mode in file:

            with open(os.path.join(save_path, file), "rb") as fp:  # Unpickling
                network_indices_correct = pickle.load(fp)
                networks_indices_correct.append(network_indices_correct)
    return networks_indices_correct


# 5. compute training metrics, like indices agreement and correlate them with dataset metrics
if args.visualize_results:
    print("Calculate agreement and other training metrics")
    print("Loading networks_indices_correct ...")

    vis_save_path = os.path.join(save_path, args.agreement_type)

    if not os.path.exists(vis_save_path):
        os.makedirs(vis_save_path)

    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(vis_save_path, "output.pdf"))

    for mode in ['train', 'test']:
        networks_indices_correct = load_indices_correct(save_path, data_mode=mode,
                                                        indices_type='network_indices_correct')

        tm = TrainingMetrics()

        # calculate
        print("Calculating ...")

        if not args.randomize_labels and not os.path.exists(os.path.join(save_path, 'random_labels.csv')):
            dataset_labels = m.evaluation['labels']
        else:
            dataset_labels = pd.read_csv(os.path.join(save_path, 'random_labels.csv'))
            dataset_labels.set_index('img_paths', inplace=True)
            dataset_labels = dataset_labels['labels']

        if args.multilabel:
            # transform multilabel prediction (with 0 or 1 for every class of an image)
            # into the single label form:
            # 1.exact match, 2. learning the same class, e.g. (0,1,0,0) for GT (0,1,1,0)
            # 3. learning at least one correct class

            labels_one_hot = labels2one_hot(dataset_labels, dataset.num_classes)
            labels_one_hot = pd.DataFrame(labels_one_hot.items(), columns=['img_paths', 'labels'])
            labels_one_hot.set_index('img_paths', inplace=True)
            labels_one_hot = labels_one_hot['labels']

            instance_agreement, lower_bound, mean_accuracy, std_accuracy, agreement_indices = \
                tm.calc_agreement_multilabel(networks_indices_correct, labels_one_hot, args.agreement_type)

        else:
            instance_agreement, lower_bound, mean_accuracy, std_accuracy, agreement_indices = \
                tm.calc_agreement(networks_indices_correct, dataset_labels, args.save_prob_vector)
            label_agreement = tm.calc_agreement_labels(agreement_indices, dataset_labels)

        # different metrics
        entropy = tm.calc_metric(agreement_indices, m.evaluation, 'entropy')
        segcount = tm.calc_metric(agreement_indices, m.evaluation, 'segcount')
        edge_strength = tm.calc_metric(agreement_indices, m.evaluation, 'edge_strength')
        freq_biggest_coeff = tm.calc_metric(agreement_indices, m.evaluation, 'freq_biggest_coeff')
        freq_coeff_percentage = tm.calc_metric(agreement_indices, m.evaluation, 'freq_coeff_percentage')
        if args.dataset == 'CIFAR10' and mode == 'test':
            # predictive entropy of the human uncertainty
            human_uncertainty = tm.calc_metric(agreement_indices, m.evaluation, 'human_uncertainty')
        if args.dataset == 'VOCDetection':
            # human response time
            img_difficulty = tm.calc_metric(agreement_indices, m.evaluation, 'img_difficulty')
            obj_sizes = tm.calc_metric(agreement_indices, m.evaluation, 'obj_sizes')
            num_instances = tm.calc_metric(agreement_indices, m.evaluation, 'num_instances')
        if args.dataset == 'KTH_TIPS':
            img_rotation = tm.calc_metric_distinct(networks_indices_correct,
                                                   agreement_indices, m.evaluation, 'rotation')
            img_illumination = tm.calc_metric_distinct(networks_indices_correct,
                                                       agreement_indices, m.evaluation, 'illumination')
            img_scale = tm.calc_metric(agreement_indices, m.evaluation, 'scale')

        # visualize
        title = dataset.name + ' ' + args.architecture + ' ' + mode
        vis.visualize_instance_agreement(instance_agreement, lower_bound, mean_accuracy,
                                         std_accuracy,
                                         title=title,
                                         save_path=vis_save_path, pdf=pdf)
        if not args.multilabel:
            vis.visualize_label_agreement_distribution(label_agreement, dataset.num_classes,
                                                   idx_to_label=dataset.idx_to_class,
                                                   title=title,
                                                   save_path=vis_save_path, pdf=pdf)
        vis.visualize_metric(entropy, 'Entropy', 'Bits', instance_agreement,
                             lower_bound, mean_accuracy, std_accuracy,
                              title=title,
                              save_path=vis_save_path, pdf=pdf)
        vis.visualize_metric(segcount, 'Segment count', 'Segment count',
                             instance_agreement, lower_bound, mean_accuracy, std_accuracy,
                             title=title,
                             save_path=vis_save_path, pdf=pdf)
        vis.visualize_metric(edge_strength, 'Edge strengths', 'Summed edge strengths',
                             instance_agreement, lower_bound, mean_accuracy, std_accuracy,
                             title=title,
                             save_path=vis_save_path, pdf=pdf)
        vis.visualize_metric(freq_biggest_coeff, 'Frequency biggest coeff', 'Number',
                             instance_agreement, lower_bound, mean_accuracy, std_accuracy,
                             title=title,
                             save_path=vis_save_path, pdf=pdf)
        vis.visualize_metric(freq_coeff_percentage, 'Frequency % coeff needed',
                             '% of coeff (99.98% energy)',
                             instance_agreement, lower_bound, mean_accuracy, std_accuracy,
                             title=title,
                             save_path=vis_save_path, pdf=pdf)

        if args.dataset == 'CIFAR10' and mode == 'test':
            vis.visualize_metric(human_uncertainty, 'Pred. entropy of human uncertainty',
                                 'Bits',
                                 instance_agreement, lower_bound, mean_accuracy, std_accuracy,
                                 title=title,
                                 save_path=vis_save_path, pdf=pdf)

        if args.dataset == 'VOCDetection':
            vis.visualize_metric(img_difficulty, 'Image difficulty',
                                 'Human response t. in sec.',
                                 instance_agreement, lower_bound, mean_accuracy, std_accuracy,
                                 title=title,
                                 save_path=vis_save_path, pdf=pdf)
            vis.visualize_metric(obj_sizes, 'Object sizes',
                                 'Bounding box area',
                                 instance_agreement, lower_bound, mean_accuracy, std_accuracy,
                                 title=title,
                                 save_path=vis_save_path, pdf=pdf)
            vis.visualize_metric(num_instances, 'Number of instances',
                                 'Number of instances',
                                 instance_agreement, lower_bound, mean_accuracy, std_accuracy,
                                 title=title,
                                 save_path=vis_save_path, pdf=pdf)

        if args.dataset == 'KTH_TIPS':
            vis.visualize_metric_distinct(img_rotation, 'Image rotation',
                                 'Percent rotation', [dataset.rotation_marker, dataset.rotation_type],
                                 instance_agreement, lower_bound, mean_accuracy, std_accuracy,
                                 title=title,
                                 save_path=vis_save_path, pdf=pdf)
            vis.visualize_metric_distinct(img_illumination, 'Image illumination',
                                 'Percent illumination', [dataset.illumination_marker, dataset.illumination_type],
                                 instance_agreement, lower_bound, mean_accuracy, std_accuracy,
                                 title=title,
                                 save_path=vis_save_path, pdf=pdf)
            vis.visualize_metric(img_scale, 'Image scale',
                                 'Average scale',
                                 instance_agreement, lower_bound, mean_accuracy, std_accuracy,
                                 title=title,
                                 save_path=vis_save_path, pdf=pdf)
    pdf.close()
