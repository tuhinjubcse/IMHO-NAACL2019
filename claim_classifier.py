from fastai.text import *
import html
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report,f1_score
from sklearn.model_selection import KFold
import torch



torch.cuda.set_device(0)
re1 = re.compile(r'  +')
         
def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))
    
def get_texts(df, n_lbls=1): 
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = list(texts.apply(fixup).values)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)

def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels

BOS = 'xbos'    # beginning-of-sentence tag# begin 
FLD = 'xfld'  # data field tag

PATH=Path('./')


CLAS_PATH=Path('./arg_class/')
CLAS_PATH.mkdir(exist_ok=True)

LM_PATH=Path('./argument_lm/')
LM_PATH.mkdir(exist_ok=True)

train_data = []
train_labels = []
test_data = []
test_labels = []


for line in open('train.tsv'):
        line = line.strip().split('\t')
        if len(line)<2:
                continue
        train_data.append(line[0])
        train_labels.append(int(line[1]))

for line in open('test.tsv'):
	line = line.strip().split('\t')
	if len(line)<2:
		continue
	test_data.append(line[0])
	test_labels.append(int(line[1]))



train_data = np.asarray(text)
train_labels = np.asarray(labels)
test_data = np.asarray(text1)
test_labels = np.asarray(labels1)
for j in range(0,1):
	
	X_train, y_train = train_data, train_labels
	X_test, y_test = test_data, test_labels
	col_names = ['labels','text']
	df_trn = pd.DataFrame({'text':X_train, 'labels':y_train}, columns=col_names)
	df_val = pd.DataFrame({'text':X_test, 'labels':y_test}, columns=col_names)
	df_trn.to_csv(CLAS_PATH/'train.csv', header=False, index=False)
	df_val.to_csv(CLAS_PATH/'test.csv', header=False, index=False)

	chunksize=24000

	df_trn = pd.read_csv(CLAS_PATH/'train.csv', header=None, chunksize=chunksize)
	df_val = pd.read_csv(CLAS_PATH/'test.csv', header=None, chunksize=chunksize)

	tok_trn, trn_labels = get_all(df_trn, 1)
	tok_val, val_labels = get_all(df_val, 1)

	(CLAS_PATH/'tmp').mkdir(exist_ok=True)

	np.save(CLAS_PATH/'tmp'/'tok_trn.npy', tok_trn)
	np.save(CLAS_PATH/'tmp'/'tok_val.npy', tok_val)

	np.save(CLAS_PATH/'tmp'/'trn_labels.npy', trn_labels)
	np.save(CLAS_PATH/'tmp'/'val_labels.npy', val_labels)


	tok_trn = np.load(CLAS_PATH/'tmp'/'tok_trn.npy')
	tok_val = np.load(CLAS_PATH/'tmp'/'tok_val.npy')

	itos = pickle.load((LM_PATH/'tmp'/'itos.pkl').open('rb'))
	stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
	print(len(itos))

	trn_clas = np.array([[stoi[o] for o in p] for p in tok_trn])
	val_clas = np.array([[stoi[o] for o in p] for p in tok_val])

	np.save(CLAS_PATH/'tmp'/'trn_ids.npy', trn_clas)
	np.save(CLAS_PATH/'tmp'/'val_ids.npy', val_clas)

	trn_clas = np.load(CLAS_PATH/'tmp'/'trn_ids.npy')
	val_clas = np.load(CLAS_PATH/'tmp'/'val_ids.npy')

	trn_labels = np.squeeze(np.load(CLAS_PATH/'tmp'/'trn_labels.npy'))
	val_labels = np.squeeze(np.load(CLAS_PATH/'tmp'/'val_labels.npy'))

	bptt,em_sz,nh,nl = 70,400,1150,3
	vs = len(itos)
	opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
	bs = 64

	min_lbl = trn_labels.min()
	trn_labels -= min_lbl
	val_labels -= min_lbl
	c=int(trn_labels.max())+1
	print(min_lbl,trn_labels.max(),c)

	trn_ds = TextDataset(trn_clas, trn_labels)
	val_ds = TextDataset(val_clas, val_labels)
	trn_samp = SortishSampler(trn_clas, key=lambda x: len(trn_clas[x]), bs=bs//2)
	val_samp = SortSampler(val_clas, key=lambda x: len(val_clas[x]))
	trn_dl = DataLoader(trn_ds, bs//2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
	val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
	md = ModelData(PATH, trn_dl, val_dl)
	print(val_samp)
	val_lbls_sampled = val_labels[list(val_samp)]


	dps = np.array([0.4, 0.5, 0.05, 0.3, 0.1])
	dps = np.array([0.4,0.5,0.05,0.3,0.4])*0.5
	m = get_rnn_classifier(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
          layers=[em_sz*3, 50, c], drops=[dps[4], 0.1],
          dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])

	opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
	learn = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
	learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
	learn.clip=.25
	learn.metrics = [accuracy]
	lr=3e-4
	lrm = 2.6
	lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr])

	lrs=np.array([1e-4,1e-4,1e-4,1e-3,1e-2])

	wd = 1e-7
	wd = 0
	learn.load_encoder('lm1_enc')
	learn.freeze_to(-1)
	learn.lr_find(lrs/1000)
	learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))
	learn.save('clas_0')
	learn.load('clas_0')
	learn.freeze_to(-2)
	learn.fit(lrs, 1, wds=wd, cycle_len=1, use_clr=(8,3))
	learn.save('clas_1')
	learn.load('clas_1')
	learn.unfreeze()
	learn.fit(lrs, 1, wds=wd, cycle_len=10, use_clr=(32,10))
	predictions = np.argmax(learn.predict(), axis=1)	
	val_f1 = f1_score(val_lbls_sampled, predictions, average='macro')
	print(classification_report(val_lbls_sampled, predictions))
