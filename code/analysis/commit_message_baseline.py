import numpy as np, pandas as pd, re, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack
D=sys.argv[2] if len(sys.argv)>2 else 'results/matched_sampling/task1/task1_b20_nometa/'
msg=pd.read_csv('data/commit_messages/messages_clean.tsv',sep='\t',dtype=str,keep_default_na=False,quoting=3)
M=dict(zip(msg.hash,msg.msg))
SEC=re.compile(r'cve|cwe|secur|vulnerab|exploit|overflow|xss|inject|csrf|sanitiz|sanitis|dos\b|denial|crash|use[- ]after|null pointer|out[- ]of[- ]bounds|oob|leak|auth|privilege|escap|malicious|attack|unsafe|fuzz|oss-fuzz|heap|stack|integer|bounds|valid',re.I)
def lg(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def vec():
    return (TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=50000,sublinear_tf=True,lowercase=True),
            TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,max_features=50000,sublinear_tf=True))
def fit_text(tr_txt,tr_y):
    v1,v2=vec(); X=hstack([v1.fit_transform(tr_txt),v2.fit_transform(tr_txt)]).tocsr()
    clf=LogisticRegression(C=4,max_iter=3000,class_weight='balanced').fit(X,tr_y)
    return lambda t: clf.decision_function(hstack([v1.transform(t),v2.transform(t)]).tocsr())
arch=sys.argv[1] if len(sys.argv)>1 else 'transformer_causal'
rows=[]
for s in [42]:
    p=np.load(D+f'preds_{arch}_b20_s{s}_nometa.npz',allow_pickle=True)
    vh=p['val_hash'].astype(str); vy=p['val_y'].astype(int); vr=p['val_repo'].astype(str); vp=p['val_pred']
    keep=np.array([h in M for h in vh]); print('val coverage',keep.mean(),keep.sum())
    vh,vy,vr,vp=vh[keep],vy[keep],vr[keep],vp[keep]; vt=[M[h] for h in vh]
    # OOF text on val
    oof=np.zeros(len(vy))
    for tr,te in GroupKFold(10).split(vt,vy,vr):
        f=fit_text([vt[i] for i in tr],vy[tr]); oof[te]=f([vt[i] for i in te])
    print('val OOF text AUC',roc_auc_score(vy,oof),'beh',roc_auc_score(vy,vp))
    st=LogisticRegression().fit(np.c_[lg(vp),oof],vy); print('stack coef',st.coef_)
    ftxt=fit_text(vt,vy)
    sets={'internal':(p['test_hash'],p['test_y'],p['test_pred'])}
    for e in ['near','unseen']:
        q=np.load(D+f'ext_{e}_{arch}_b20_s{s}_nometa.npz',allow_pickle=True)
        sets[e]=(q['hash'],q['labels'],q['scores'])
    for name,(h,y,b) in sets.items():
        h=h.astype(str); y=y.astype(int); k=np.array([x in M for x in h])
        h,y,b=h[k],y[k],b[k]; t=[M[x] for x in h]; tx=ftxt(t); cb=st.decision_function(np.c_[lg(b),tx])
        unin=np.array([not SEC.search(x) for x in t]); nocve=np.array([not re.search(r'cve|cwe|secur|vulnerab',x,re.I) for x in t])
        def a(m,s_): 
            return roc_auc_score(y[m],s_[m]) if len(set(y[m]))>1 else np.nan
        al=np.ones(len(y),bool)
        for sub,m in [('all',al),('no-sec-words',unin),('no-CVE/security',nocve)]:
            rows.append(dict(set=name,subset=sub,n=m.sum(),pos=y[m].sum(),cov=k.mean(),text=a(m,tx),behav=a(m,b),comb=a(m,cb)))
r=pd.DataFrame(rows); pd.set_option('display.width',200); print(r.round(3).to_string())
r.to_csv(f'results/commit_message_baseline/res_{arch}.csv',index=False)
# bootstrap comb-text and comb-behav on 'all' and 'no-sec-words'
rng=np.random.default_rng(0)
for name,(h,y,b) in sets.items():
    h=h.astype(str); y=y.astype(int); k=np.array([x in M for x in h]); h,y,b=h[k],y[k],b[k]; t=[M[x] for x in h]
    tx=ftxt(t); cb=st.decision_function(np.c_[lg(b),tx]); unin=np.array([not SEC.search(x) for x in t])
    for sub,m in [('all',np.ones(len(y),bool)),('no-sec-words',unin)]:
        yy,tt,bb,cc=y[m],tx[m],b[m],cb[m]; d1=[];d2=[]
        for _ in range(2000):
            i=rng.integers(0,len(yy),len(yy))
            if len(set(yy[i]))<2: continue
            d1.append(roc_auc_score(yy[i],cc[i])-roc_auc_score(yy[i],tt[i])); d2.append(roc_auc_score(yy[i],cc[i])-roc_auc_score(yy[i],bb[i]))
        print(name,sub,'comb-text',np.round(np.percentile(d1,[2.5,50,97.5]),3),'comb-beh',np.round(np.percentile(d2,[2.5,50,97.5]),3))
