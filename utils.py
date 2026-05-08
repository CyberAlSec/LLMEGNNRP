import ast
import os
import random
import re
import os.path as osp
import numpy as np
import time
import datetime
import pytz
import torch
from llvmlite.binding import FeatureMap
from torch import Tensor, nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import scipy.sparse as sp

from data_utils.load import load_data
FeatureMap = {
    "E5": ["infloat/e5-large",1024],
    "TAPE":['/home/mayuhang/文档/LLMEGNNRP/model/deberta-base',768],
    "ModernBert": ['answerdotai/ModernBERT-large',1024],
    "Linq": ["Linq-AI-Research/Linq-Embed-Mistral",None],
    "Llama":["meta-llama/Llama-2-7b-hf",None]
}
CLASS_MAP ={"cora":{x: i for i, x in enumerate(['CaseBased', 'GeneticAlgorithms', 'NeuralNetworks',
                                         'ProbabilisticMethods', 'ReinforcementLearning', 'RuleLearning', 'Theory'])},
                 "pubmed": {x: i for i, x in enumerate(['Type1diabetes',"Type2diabetes","Experimentallyinduceddiabetes"])},
                 "arxiv_2023": {x:i for i,x in enumerate(['cs.AI','cs.AR','cs.CC','cs.CE','cs.CG','cs.CL','cs.CR','cs.CV','cs.CY','cs.DB','cs.DC','cs.DL','cs.DM','cs.DS','cs.ET','cs.FL','cs.GL','cs.GR','cs.GT','cs.HC','cs.IR','cs.IT','cs.LG','cs.LO','cs.MA','cs.MM','cs.MS','cs.NA','cs.NE','cs.NI','cs.OH','cs.OS','cs.PF','cs.PL','cs.RO','cs.SC','cs.SD','cs.SE','cs.SI','cs.SY'])},
                 "ogbn-products":{x:i for i,x in enumerate(['Home&Kitchen', 'Health&PersonalCare', 'Beauty','Sports&Outdoors', 'Books','Patio,Lawn&Garden','Toys&Games','CDs&Vinyl','CellPhones&Accessories','Grocery&GourmetFood','Arts,Crafts&Sewing','Clothing,Shoes&Jewelry','Electronics','Movies&TV','Software','VideoGames','Automotive','PetSupplies','OfficeProducts','Industrial&Scientific','MusicalInstruments','Tools&HomeImprovement','MagazineSubscriptions','BabyProducts','NAN','Appliances','Kitchen&Dining','Collectibles&FineArt','AllBeauty','LuxuryBeauty','AmazonFashion','Computers','AllElectronics','PurchaseCircles','MP3Players&Accessories','GiftCards','Office&SchoolSupplies','HomeImprovement','Camera&Photo','GPS&Navigation','DigitalMusic','CarElectronics','Baby','KindleStore','KindleApps','Furniture&Decor'])}
                 }

def init_random_state(seed=0):
    # Libraries using GPU should be imported after specifying GPU-ID
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    row,col,_ = torch_sparse.coo()
    value = torch.ones(row.size(0),dtype=torch.float32)
    sp_matrix = sp.coo_matrix((value,(row,col)),shape=(torch_sparse.sizes()[0],torch_sparse.sizes()[1]))
    return sp_matrix

def get_model_name(feature):
    return FeatureMap[feature][0],FeatureMap[feature][1]

def add_loop_sparse(adj, fill_value=1):
    # make identify sparse tensor
    row = torch.range(0, int(adj.shape[0]-1), dtype=torch.int64)
    i = torch.stack((row, row), dim=0)
    v = torch.ones(adj.shape[0], dtype=torch.float32)
    shape = adj.shape
    I_n = torch.sparse.FloatTensor(i, v, shape)
    return adj + I_n
import torch

def dedup_edge_index(edge_index: torch.Tensor,
                     undirected: bool = True,
                     remove_self_loops: bool = False) -> torch.Tensor:
    """
    对 edge_index ([2, E]) 去重，确保每条边只出现一次。
    如果 undirected=True，则 (u,v) 与 (v,u) 视为同一条边。

    返回:
      去重后的 edge_index ([2, E'])
    """
    assert edge_index.dim() == 2 and edge_index.size(0) == 2, \
        "edge_index must be shape [2, E]"

    src, dst = edge_index[0], edge_index[1]

    if undirected:
        # 把 (u,v) 无向归一化为 (min,max)
        a = torch.minimum(src, dst)
        b = torch.maximum(src, dst)
        pairs = torch.stack([a, b], dim=1)  # [E,2]
    else:
        pairs = torch.stack([src, dst], dim=1)

    # 去重
    pairs = torch.unique(pairs, dim=0)

    # 可选：移除自环
    if remove_self_loops:
        mask = pairs[:, 0] != pairs[:, 1]
        pairs = pairs[mask]

    if undirected:
        # 为了返回完整的无向 edge_index（即包含双向边）
        rev = pairs[:, [1, 0]]
        pairs = torch.cat([pairs, rev], dim=0)

    # 转置为 [2, E']
    edge_index_dedup = pairs.t().contiguous()
    return edge_index_dedup


class Evaluator:
    def __init__(self, name):
        self.name = name

    def eval(self, input_dict):
        y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            acc_list.append(float(np.sum(correct))/len(correct))

        return {'acc': sum(acc_list)/len(acc_list)}


def classification_margin_list(output,labels):
    probs = torch.exp(output)
    # labels_test = labels[test_mask].to(self.device)
    true_probs = probs.gather(1, labels.view(-1, 1)).squeeze(1)
    probs_test = probs.clone()
    probs_test[torch.arange(probs.shape[0]), labels] = float('-inf')
    probs_max, _ = torch.max(probs_test, dim=1)
    margins = true_probs - probs_max
    margin = torch.mean(margins).item()

    return margin

def batched_data(inputs, batch_size):
    return [inputs[i:i+batch_size] for i in range(0, len(inputs), batch_size)]

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def generate_pred(data, data_name,attack_name):
    root =get_project_root()
    if attack_name is not None:
        adv = torch.load(f"{root}/attack/data/{data_name}/text/{data_name}_{attack_name}_orig.pt")
        out_file = f"{root}/attack/data/TAPE/{data_name}/{data_name}_{attack_name}_pred.pt"
    else:
        out_file = f"{root}/attack/data/TAPE/{data_name}/{data_name}_clean_pred.pt"

    class_map = CLASS_MAP[data_name]

    result = []
    for idx ,item in enumerate(data):
        temp  = item.split("\n\n")[0]
        pred_strs = re.sub(r'\d+\)','',temp).split(",")
        pred = []
        for pred_str in pred_strs:
            index = class_map.get(pred_str.replace(" ",""))
            if not index == None:
                pred.append(index)
        if len(pred) == 0:
            pred = [-1]
        if attack_name is not None:
            result.append({"idx":adv[idx]["idx"],"pred":pred})
        else:
            result.append(pred)
    torch.save(result,out_file)
    print(f"{data_name}_{attack_name} success save")


def get_e5_large_embedding(texts, device, dataset_name = 'cora', batch_size = 64, cache_out = '/tmp', update = True):
    output_path = osp.join(cache_out, dataset_name + "_e5_embedding.pt")
    if osp.exists(output_path) and not update:
        return torch.load(output_path, map_location='cpu')
    texts = ["query: " + x for x in texts]
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large', cache_dir='/tmp')
    model = AutoModel.from_pretrained('intfloat/e5-large', cache_dir='/tmp').to(device)
    # Tokenize the input texts
    output = []
    with torch.no_grad():
        for batch in tqdm(batched_data(texts, batch_size)):
            batch_dict = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
            outputs = model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            output.append(embeddings.cpu())
            del batch_dict
    output = torch.cat(output, dim = 0)
    return output


def knowledge_augmentation(original_texts: list, augment_knowledge: list, strategy: str = "back"):
    """
        If inplace, insert knowledge into original_texts.
        If back, insert it to back
        If separate, separate two texts, encode each one
    """
    total_texts = []
    total_know = []
    for texts, know_t in zip(original_texts, augment_knowledge):
        # know = know_t[0]
        know = know_t
        dict_str = know
        know_str = ""
        t = texts
        if know is None:
            print(len(total_texts))
            continue
        try:
            first_curly = know.index('{')
            second_curly = know.index('}')
            dict_str = know[first_curly:second_curly + 1]
            know_dict = ast.literal_eval(dict_str)
            for key, value in know_dict.items():
                if strategy == "back":
                    t += f" {value} "
                    know_str = dict_str
                elif strategy == "separate":
                    know_str += f" {value};"
                    # total_know.append(know_str)
                    # total_texts.append(texts)
                elif strategy == "inplace":
                    pos = texts.find(key) + len(key)
                    t = texts[:pos] + f" ({value}) " + texts[pos:]
                    know_str = dict_str
                    # total_texts.append(texts)
        except Exception:
            ## format error
            if strategy == "back" or strategy == "inplace":
                t += dict_str
                # total_texts.append(texts)
                know_str = dict_str
            elif strategy == "separate":
                know_str += dict_str

        total_know.append(know_str)
        total_texts.append(t)
    return total_texts, total_know


def get_project_root():
    current_file = os.path.abspath(__file__)

    project_root = os.path.dirname(current_file)

    while not os.path.exists(os.path.join(project_root, 'README.md')):
        parent = os.path.dirname(project_root)
        if parent == project_root:
            return None
        project_root = parent

    return project_root


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path):
        return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file


# * ============================= Time Related =============================


def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(
            f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.')
        return ret

    return wrapper

class BOW(nn.Module):
    def __init__(self, dataset, vocab_size, device):
        super(BOW, self).__init__()
        training_txt = [
            txt
            for i, txt in enumerate(dataset.text)
            if i in dataset.train_id
        ]
        self.device = device

        self.counter = CountVectorizer()
        self.counter.fit(dataset.text)
        self.tokenizer = self.counter.build_tokenizer()
        self.vocab = self.counter.vocabulary_

        feature = self.counter.transform(training_txt)
        counts = torch.LongTensor(feature.toarray().sum(0))
        freq_idx = counts.argsort(descending=True)
        inv_vocab = {v: k for k, v in self.vocab.items()}
        high_freq_words = [inv_vocab[i.item()].lower() for i in freq_idx][:vocab_size]

        total_vocab = len(self.vocab) + 1
        self.embedding = nn.Embedding(total_vocab, vocab_size)
        self.embedding.weight.data.zero_()
        for idx_in_list, word in enumerate(high_freq_words):
            raw_idx = self.vocab.get(word, None)
            if raw_idx is not None:
                self.embedding.weight.data[raw_idx + 1, idx_in_list] = 1.0

        self.vocab = {word: idx + 1 for word, idx in self.vocab.items()}
        self.embedding.to(self.device)

    def generate(self, input_ids=None, perturbed=None):

        if perturbed is None:
            # lookup embedding -> shape [batch_size, seq_len, vocab_size]
            embed = self.embedding(input_ids.to(self.device))
        else:
            embed = perturbed
        bow_counts = embed.sum(dim=1)

        bow_binary = (bow_counts > 0).to(bow_counts.dtype)
        return bow_binary

    def text_to_ids(self, texts):

        batch_ids = []
        for txt in texts:
            tokens = self.tokenizer(txt.lower())
            ids = [self.vocab.get(tok, 0) for tok in tokens]
            batch_ids.append(torch.LongTensor(ids))

        max_len = max(t.size(0) for t in batch_ids)
        padded = []
        for t in batch_ids:
            if t.size(0) < max_len:
                pad = torch.zeros(max_len - t.size(0), dtype=torch.long)
                t = torch.cat([t, pad], dim=0)
            padded.append(t)
        return torch.stack(padded, dim=0)

    def embed_texts(self, texts):

        ids = self.text_to_ids(texts)
        return self.generate(input_ids=ids)

class TFIDFExtractor(nn.Module):
    def __init__(self, dataset, vocab_size: int, device: torch.device):

        super(TFIDFExtractor, self).__init__()
        self.device = device


        train_texts = [txt for idx, txt in enumerate(dataset.text) if idx in dataset.train_id]


        self.vectorizer = TfidfVectorizer(
            max_features=vocab_size,
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b"
        )
        self.vectorizer.fit(train_texts)


        self.vocab = {term: idx for term, idx in self.vectorizer.vocabulary_.items()}

        self.feature_names = self.vectorizer.get_feature_names_out().tolist()

    def embed_texts(self, texts: list) -> torch.FloatTensor:

        tfidf_matrix = self.vectorizer.transform(texts)  # scipy.sparse matrix

        dense = tfidf_matrix.toarray()
        tensor = torch.from_numpy(dense).float().to(self.device)
        return tensor

    def forward(self, texts: list) -> torch.FloatTensor:

        return self.embed_texts(texts)

class Word2VecEncoder(nn.Module):
    """
    Memory-optimized Word2Vec text encoder.
    - Trains Word2Vec on CPU.
    - Uses sparse, half-precision, frozen embeddings.
    - Only moves embedding to device during forward with no_grad.
    """
    def __init__(
        self,
        dataset,
        vector_size: int = 64,
        window: int = 5,
        min_count: int = 5,
        sg: int = 1,
        device: torch.device = None
    ):
        super().__init__()
        self.device = device or torch.device('cpu')

        texts = [
            txt
            for i, txt in enumerate(dataset.text)
            if i in dataset.train_id
        ]
        counter = CountVectorizer()
        counter.fit(texts)
        self.tokenizer = counter.build_tokenizer()
        sentences = [self.tokenizer(doc.lower()) for doc in texts]

        self.w2v_model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            workers=4,
            epochs=10
        )

        self.vocab = {w: i+1 for w, i in self.w2v_model.wv.key_to_index.items()}
        num_embeddings = len(self.vocab) + 1

        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=vector_size,
            padding_idx=0,
            sparse=True
        )

        weights = torch.zeros(num_embeddings, vector_size)
        for word, idx in self.vocab.items():
            weights[idx] = torch.from_numpy(self.w2v_model.wv[word])
        # half precision
        weights = weights.half()
        self.embedding.weight.data.copy_(weights)

        self.embedding.weight.requires_grad_(False)

        self.embedding.to(self.device)

    def text_to_ids(self, texts) -> torch.LongTensor:

        batch_ids = []
        for txt in texts:
            toks = self.tokenizer(txt.lower())
            ids = [self.vocab.get(t, 0) for t in toks]
            batch_ids.append(torch.LongTensor(ids))


        max_len = max(t.size(0) for t in batch_ids)
        padded = []
        for t in batch_ids:
            if t.size(0) < max_len:
                pad = torch.zeros(max_len - t.size(0), dtype=torch.long)
                t = torch.cat([t, pad], dim=0)
            padded.append(t)
        return torch.stack(padded, dim=0).to(self.device)

    def generate(self, input_ids: torch.LongTensor) -> torch.Tensor:

        with torch.no_grad():

            emb = self.embedding(input_ids)    # [B, L, D]
            mask = (input_ids > 0).unsqueeze(-1)
            mask = mask.to(emb.dtype)
            summed = (emb * mask).sum(dim=1)          # [B, D]
            counts = mask.sum(dim=1).clamp(min=1)
            doc_vec = summed / counts
        return doc_vec.cpu()


    def embed_texts(self, texts) -> torch.Tensor:
        ids = self.text_to_ids(texts)
        return self.generate(ids)