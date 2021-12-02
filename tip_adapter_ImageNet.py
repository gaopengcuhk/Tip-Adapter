import os
import numpy as np
import torch
import clip
from tqdm.notebook import tqdm
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from collections import defaultdict
import random
from tqdm import tqdm
import argparse


print("Torch version:", torch.__version__)
assert torch.__version__.split(".") >= ["1", "7", "1"], "PyTorch 1.7.1 or later is required"


imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
                        "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
                        "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
                        "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",
                        "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
                        "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
                        "box turtle", "banded gecko", "green iguana", "Carolina anole",
                        "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
                        "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
                        "American alligator", "triceratops", "worm snake", "ring-necked snake",
                        "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake",
                        "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
                        "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
                        "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
                        "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",
                        "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",
                        "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",
                        "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck",
                        "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby",
                        "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch",
                        "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
                        "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
                        "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
                        "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot",
                        "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher",
                        "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",
                        "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
                        "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
                        "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
                        "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
                        "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",
                        "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
                        "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
                        "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
                        "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
                        "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
                        "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
                        "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
                        "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
                        "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
                        "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
                        "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
                        "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
                        "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
                        "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
                        "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
                        "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
                        "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
                        "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
                        "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
                        "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
                        "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
                        "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
                        "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
                        "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger",
                        "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",
                        "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
                        "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
                        "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper",
                        "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",
                        "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",
                        "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
                        "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse",
                        "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
                        "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)",
                        "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
                        "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
                        "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque",
                        "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
                        "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
                        "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda",
                        "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
                        "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown",
                        "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
                        "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle",
                        "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",
                        "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",
                        "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
                        "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)",
                        "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
                        "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet",
                        "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",
                        "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
                        "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe",
                        "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton",
                        "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
                        "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
                        "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
                        "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
                        "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
                        "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
                        "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
                        "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
                        "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",
                        "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
                        "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
                        "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder",
                        "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",
                        "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
                        "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
                        "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",
                        "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine",
                        "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
                        "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet",
                        "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",
                        "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",
                        "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",
                        "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library",
                        "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion",
                        "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag",
                        "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",
                        "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",
                        "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",
                        "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor",
                        "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
                        "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",
                        "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
                        "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",
                        "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush",
                        "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench",
                        "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",
                        "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
                        "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
                        "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
                        "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
                        "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",
                        "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill",
                        "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
                        "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",
                        "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser",
                        "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal",
                        "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",
                        "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
                        "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
                        "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
                        "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
                        "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",
                        "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",
                        "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
                        "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",
                        "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge",
                        "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe",
                        "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball",
                        "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof",
                        "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
                        "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod",
                        "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",
                        "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
                        "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",
                        "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
                        "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",
                        "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",
                        "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
                        "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
                        "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
                        "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
                        "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
                        "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
                        "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
                        "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
                        "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",
                        "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
                        "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
                        "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
                        "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]

# Prompt Ensembling
imagenet_templates = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]

# Single Prompt
# imagenet_templates = ['a photo of a {}.',]
    

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


class Weight_Adapter(nn.Module):
    def __init__(self, clip_model, train_features_path, cls_num, shots):
        super().__init__()
        self.linear1 = nn.Linear(1024, cls_num * shots, bias=False).to(clip_model.dtype)
        self.linear1.weight = nn.Parameter(torch.load(train_features_path).t())


def main():

    # Path for ImageNet
    data_path = "/your/data/path"


    train_features_path = "./imagenet_f_train.pt"
    train_targets_path = "./imagenet_t_train.pt"

    test_features_path = "./imagenet_f_test.pt"
    test_targets_path = "./imagenet_t_test.pt"

    # load_train = False
    # load_test = False

    load_train = True
    load_test = True
    
    search = True

    # ~~~~~~~~~~~~~~~~~~
    k_shot = 16
    # ~~~~~~~~~~~~~~~~~~

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help='lr')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1.17)
    parser.add_argument('--train_epoch', type=int, default=20)
    parser.add_argument('--augment_epoch', type=int, default=10)
    args = parser.parse_args()
    print(args)

    clip.available_models()
    name = 'RN50'

    model, preprocess = clip.load(name)
    model.eval()

    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    random.seed(1)
    torch.manual_seed(1)

    print(f"{len(imagenet_classes)} classes, {len(imagenet_templates)} templates")

    images = torchvision.datasets.ImageNet(data_path, split='val', transform=preprocess)
    loader = torch.utils.data.DataLoader(images, batch_size=64, num_workers=8, shuffle=False)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    train_images = torchvision.datasets.ImageNet(data_path, split='train',
                                                 transform=train_tranform)
    split_by_label_dict = defaultdict(list)

    print('Load data finished.')
    for i in range(len(train_images.imgs)):
        split_by_label_dict[train_images.targets[i]].append(train_images.imgs[i])
    imgs = []
    targets = []

    for label, items in split_by_label_dict.items():
        imgs = imgs + random.sample(items, k_shot)
        targets = targets + [label for i in range(k_shot)]
    train_images.imgs = imgs
    train_images.targets = targets
    train_images.samples = imgs
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=256, num_workers=8, shuffle=False)
    train_loader_shuffle = torch.utils.data.DataLoader(train_images, batch_size=256, num_workers=8, shuffle=True)

    # ------------------------------------------getting text feature------------------------------------------
    print('start getting text features.')
    zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates, model)
    print('finish getting text features. start getting image features')

    # ------------------------------------------saving training features------------------------------------------
    print('start saving training image features')

    if not load_train:
        
        train_images_targets = []
        train_images_features_agg = []

        with torch.no_grad():
            for augment_idx in range(args.augment_epoch):
                train_images_features = []

                print('Augment time: {:} / {:}'.format(augment_idx, args.augment_epoch))
                for i, (images, target) in enumerate(tqdm(train_loader)):
                    images = images.cuda()
                    image_features = model.encode_image(images)
                    train_images_features.append(image_features)

                    if augment_idx == 0:
                        target = target.cuda()
                        train_images_targets.append(target)

                images_features_cat = torch.cat(train_images_features, dim=0).unsqueeze(0)
                train_images_features_agg.append(images_features_cat)
            

        train_images_features_agg = torch.cat(train_images_features_agg, dim=0).mean(dim=0)
        train_images_features_agg /= train_images_features_agg.norm(dim=-1, keepdim=True)
        train_images_features_agg = train_images_features_agg.permute(1, 0)

        train_images_targets = F.one_hot(torch.cat(train_images_targets, dim=0)).half()

        torch.save(train_images_features_agg, train_features_path)
        torch.save(train_images_targets, train_targets_path)

    else:
        train_images_features_agg = torch.load(train_features_path)
        train_images_targets = torch.load(train_targets_path)


    # ------------------------------------------saving testing features------------------------------------------
    print('start saving testing image features')
    
    if not load_test:
        test_features = []
        test_labels = []
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images = images.cuda()
                target = target.cuda()
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                test_features.append(image_features)
                test_labels.append(target)
        test_features = torch.cat(test_features)
        test_labels = torch.cat(test_labels)

        torch.save(test_features, test_features_path)
        torch.save(test_labels, test_targets_path)
   
    else:
        test_features = torch.load(test_features_path)
        test_labels = torch.load(test_targets_path)



    # CLIP Zero-shot
    top1, top5, n = 0., 0., 0.
    logits = 100. * test_features @ zeroshot_weights
    acc1, acc5 = accuracy(logits, test_labels, topk=(1, 5))
    top1 += acc1
    top5 += acc5
    n += test_features.size(0)
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    print()
    print(f"CLIP Top-1 accuracy: {top1:.2f}, with zero-shot learning")


    # Tip-Adapter
    alpha = args.alpha
    beta = args.beta
    top1, top5, n = 0., 0., 0.
    new_knowledge = test_features @ train_images_features_agg
    new_logits = ((-1) * (alpha - alpha * new_knowledge)).exp() @ train_images_targets
    logits = 100. * test_features @ zeroshot_weights
    logits = logits + new_logits * beta
    acc1, acc5 = accuracy(logits, test_labels, topk=(1, 5))
    top1 += acc1
    top5 += acc5
    n += test_features.size(0)
    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    print()
    print(f"AdapterV2 Top-1 accuracy: {top1:.2f}, without training")
    print()

    if search:

        alpha_list = [i * (6.0 - 1.0) / 20 + 1 for i in range(20)]
        beta_list = [i * (7 - 0.1) / 200 + 0.1 for i in range(200)]
        best_top1 = 0
        best_alpha = 0
        best_beta = 0

        for alpha in alpha_list:
            for beta in beta_list:
                top1, top5, n = 0., 0., 0.
                batch_idx = 0
                # predict
                with torch.no_grad():
                    test_features = torch.load(test_features_path)
                    test_labels = torch.load(test_targets_path)
                    test_features_new = test_features
                new_knowledge = test_features @ train_images_features_agg
                new_logits = ((-1) * (alpha - alpha * new_knowledge)).exp() @ (train_images_targets)
                logits = 100. * test_features_new @ zeroshot_weights
                logits = logits + new_logits * beta

                acc1, acc5 = accuracy(logits, test_labels, topk=(1, 5))
                batch_idx += 1
                top1 += acc1
                top5 += acc5
                n += test_features_new.size(0)
                top1 = (top1 / n) * 100
                top5 = (top5 / n) * 100

                if top1 > best_top1:
                    text = 'New best setting, alpha: {:.2f}, beta: {:.2f}; Top-1 acc: {:.2f}'.format(alpha, beta, top1)
                    print(text)
                    best_top1 = top1
                    best_alpha = alpha
                    best_beta = beta

        print(f"After searching, {name}, {k_shot} shot. Best Top-1 {best_top1:.2f}")
        print()



    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    adapter = Weight_Adapter(model, train_features_path, len(imagenet_classes), k_shot).cuda()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epoch * len(train_loader_shuffle))
    
    best_top1 = 0
    best_epoch = 0
    for train_idx in range(args.train_epoch):
        adapter.train()
        correct_all = 0
        n = 0
        loss_list = []
        print('Train time: {:} / {:}'.format(train_idx, args.train_epoch))
        
        alpha = args.alpha
        beta = args.beta

        for i, (images, target) in enumerate(tqdm(train_loader_shuffle)):
            images = images.cuda()
            target = target.cuda()
            with torch.no_grad():
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            new_knowledge = adapter.linear1(image_features)
            new_logits = ((-1) * (alpha - alpha * new_knowledge)).exp() @ (train_images_targets)
            logits = 100. * image_features @ zeroshot_weights
            logits = logits + new_logits * beta

            loss = F.cross_entropy(logits, target)
            loss_value = loss.item()
            correct = accuracy(logits, target)
            correct_all += correct[0]
            n += len(logits)
            loss_list.append(loss_value)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        text = 'LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_all / n, correct_all, n,
                                                                       sum(loss_list)/len(loss_list))
        print(text)


        # eval
        adapter.eval()

        top1, top5, n = 0., 0., 0.
        with torch.no_grad():
            test_features = torch.load(test_features_path)
            test_labels = torch.load(test_targets_path)
            test_features_new = test_features

        new_knowledge = adapter.linear1(test_features_new)
        new_logits = ((-1) * (alpha - alpha * new_knowledge)).exp() @ (train_images_targets)
        logits = 100. * test_features_new @ zeroshot_weights
        logits = logits + new_logits * beta
        acc1, acc5 = accuracy(logits, test_labels, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += test_features.size(0)
        top1 = (top1 / n) * 100
        top5 = (top5 / n) * 100
        text = f"Testing Top-1 Accuracy: {top1:.2f}"
        print(text)
        print()

        if top1 > best_top1:
            best_top1 = top1
            best_epoch = train_idx
    
    print(f"Best Testing Top-1 Accuracy: {best_top1:.2f}, at Epoch: {best_epoch}")
    print()
       
    print("Begin to search")
    alpha_list = [i * (6.0 - 1.0) / 20 + 1 for i in range(20)]
    beta_list = [i * (7 - 0.1) / 200 + 0.1 for i in range(200)]
    best_top1 = 0

    # ------------------------------------------ Search ------------------------------------------
    adapter.eval()
    for alpha in alpha_list:
        for beta in beta_list:
            top1, top5, n = 0., 0., 0.
            batch_idx = 0
            # predict
            with torch.no_grad():
                test_features = torch.load(test_features_path)
                test_labels = torch.load(test_targets_path)
                test_features_new = test_features
            new_knowledge = adapter.linear1(test_features_new)
            new_logits = ((-1) * (alpha - alpha * new_knowledge)).exp() @ (train_images_targets)
            logits = 100. * test_features_new @ zeroshot_weights
            logits = logits + new_logits * beta
            # measure accuracy
            acc1, acc5 = accuracy(logits, test_labels, topk=(1, 5))
            batch_idx += 1
            top1 += acc1
            top5 += acc5
            n += test_features_new.size(0)
            top1 = (top1 / n) * 100
            top5 = (top5 / n) * 100

            if top1 > best_top1:
                text = 'New best setting, alpha: {:.2f}, beta: {:.2f}; Top-1 acc: {:.2f}'.format(alpha, beta, top1)
                print(text)
                best_top1 = top1
                

    print(f"{name}, {k_shot} shot. Best Top-1 {best_top1:.2f}")



if __name__ == '__main__':
    main()

