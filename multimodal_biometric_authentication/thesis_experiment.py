from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'DSL-StrongPasswordData.csv'
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)


def eer_stats(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    fnr = 1 - tpr
    pos = np.argmin(np.abs(fpr - fnr))
    return float((fpr[pos] + fnr[pos]) / 2), float(thresholds[pos])


def distance_score(samples, center, spread):
    return -np.sum(np.abs(samples - center) / spread, axis=1)


def main():
    if not DATA.is_file():
        raise FileNotFoundError(f'Missing CMU CSV. Put it at: {DATA}')
    df = pd.read_csv(DATA)
    feat = [c for c in df.columns if c.startswith(('H.', 'DD.', 'UD.'))]
    if len(feat) != 31 or not {'subject','sessionIndex','rep'}.issubset(df.columns):
        raise ValueError('Unexpected CMU dataset columns. Check that you downloaded the original CSV.')
    people = sorted(df.subject.unique())
    if len(people) != 51:
        raise ValueError(f'Expected 51 CMU subjects; found {len(people)}')
    keys = []
    for sid in people:
        one = df[df.subject == sid].sort_values(['sessionIndex','rep'])
        if len(one) != 400:
            raise ValueError(f'{sid} has {len(one)} rows, expected 400')
        keys.append(one[feat].to_numpy(dtype=float))
    keys = np.asarray(keys)
    means = keys[:, :200].mean(axis=1)
    mad = np.maximum(np.abs(keys[:, :200] - means[:, None]).mean(axis=1), 1e-6)
    print('Loaded CMU:', keys.shape, '(subjects, repetitions, features)')

    #CMU benchmark unimodal experiment: 51 target accounts.
    keystroke_rows = []
    for i in range(51):
        genuine = distance_score(keys[i, 200:], means[i], mad[i])
        impostor = np.concatenate([distance_score(keys[j, :5], means[i], mad[i])
                                   for j in range(51) if j != i])
        labels = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
        e, _ = eer_stats(labels, np.r_[genuine, impostor])
        keystroke_rows.append({'subject': people[i], 'eer': e})
    pd.DataFrame(keystroke_rows).to_csv(OUT / 'keystroke_subject_eer.csv', index=False)
    print(f'CMU 51-subject mean EER: {np.mean([r["eer"] for r in keystroke_rows]):.2%}')

    faces = fetch_olivetti_faces(shuffle=False)
    if faces.data.shape != (400,4096):
        raise ValueError(f'Unexpected Olivetti shape: {faces.data.shape}')
    # Each face identity has ten images. Use the first 5 only for enrollment.
    face = faces.data.reshape(40,10,4096)
    enrollment = face[:, :5].reshape(200,4096)
    pca = PCA(n_components=50, svd_solver='randomized', random_state=42)
    pca.fit(enrollment)
    projected = pca.transform(face.reshape(400,4096)).reshape(40,10,50)
    templates = projected[:, :5].mean(axis=1)
    face_scores = -np.linalg.norm(projected[None,:,:,:] - templates[:,None,None,:], axis=-1)
    print('Loaded Olivetti:', face.shape, '; PCA components:', pca.n_components_)

    face_rows = []
    for i in range(40):
        genuine = face_scores[i, i, 5:]
        impostor = face_scores[i, np.arange(40) != i, 5:].ravel()
        y = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
        e, _ = eer_stats(y, np.r_[genuine, impostor])
        face_rows.append({'subject': i, 'eer': e})
    pd.DataFrame(face_rows).to_csv(OUT / 'face_subject_eer.csv', index=False)
    print(f'Olivetti 40-subject mean EER: {np.mean([r["eer"] for r in face_rows]):.2%}')

    #Chimeric IDs pair Olivetti ID i with CMU ID people[i].
    #Both modalities in each trial are from the SAME synthetic probe identity j.
    #Development probes and test probes have nonoverlapping indices.
    def trials(face_idx, key_idx):
        rows = []
        for target in range(40):
            for probe in range(40):
                ks = distance_score(keys[probe, key_idx], means[target], mad[target])
                fs = face_scores[target, probe, face_idx]
                for k, f in zip(ks, fs):
                    rows.append({'target':target, 'probe':probe,
                                 'genuine':int(target == probe),
                                 'key':float(k), 'face':float(f)})
        return pd.DataFrame(rows)

    dev = trials([5,6], [225,275])
    test = trials([7,8,9], [325,350,375])
    print(f'Chimeric development trials: {len(dev)}; final test trials: {len(test)}')

    #All normalization parameters are fitted to DEVELOPMENT scores only.
    def normalize(train, heldout, mode):
        a, b = [], []
        for col in ['face','key']:
            src = train[col].to_numpy()
            if mode == 'minmax':
                lo, hi = src.min(), src.max()
                ta = (src-lo) / max(hi-lo, 1e-9)
                tb = (heldout[col].to_numpy()-lo) / max(hi-lo, 1e-9)
            else:
                mu, sd = src.mean(), max(src.std(), 1e-9)
                ta = (src-mu) / sd
                tb = (heldout[col].to_numpy()-mu) / sd
                if mode == 'tanh':
                    ta = 0.5 * (np.tanh(0.01*ta)+1)
                    tb = 0.5 * (np.tanh(0.01*tb)+1)
            a.append(ta)
            b.append(tb)
        return np.column_stack(a), np.column_stack(b)

    #Tune the normalization function and fusion weight on development data.
    choices = []
    for mode in ['minmax','zscore','tanh']:
        d, _ = normalize(dev, test, mode)
        for weight in np.linspace(0,1,11):
            score = weight*d[:,0] + (1-weight)*d[:,1]
            e, th = eer_stats(dev.genuine, score)
            choices.append({'normalization':mode,'face_weight':round(float(weight),2),
                            'dev_eer':e,'dev_threshold':th})
    table = pd.DataFrame(choices).sort_values(['dev_eer','normalization','face_weight'])
    table.to_csv(OUT / 'development_weight_sweep.csv', index=False)
    best = table.iloc[0]
    mode, weight = best['normalization'], float(best['face_weight'])
    dev_norm, test_norm = normalize(dev, test, mode)
    print(f'Selected using development only: {mode}, face weight={weight:.1f}')

    models = {
        'Face only': (dev_norm[:,0],test_norm[:,0]),
        'Keystroke only': (dev_norm[:,1],test_norm[:,1]),
        'Fused': (weight*dev_norm[:,0]+(1-weight)*dev_norm[:,1],
                  weight*test_norm[:,0]+(1-weight)*test_norm[:,1])
    }
    summary = []
    plt.figure(figsize=(7,5))
    for name, (dscore, tscore) in models.items():
        dev_eer, threshold = eer_stats(dev.genuine, dscore)
        test_eer, _ = eer_stats(test.genuine, tscore)
        y = test.genuine.to_numpy(dtype=int)
        far = np.mean(tscore[y == 0] >= threshold)
        frr = np.mean(tscore[y == 1] < threshold)
        auc = roc_auc_score(y, tscore)
        summary.append({'model':name,'test_eer':test_eer,'test_auc':auc,
                        'dev_threshold':threshold,'test_far_at_dev_threshold':far,
                        'test_frr_at_dev_threshold':frr})
        fpr, tpr, _ = roc_curve(y, tscore)
        plt.plot(fpr,tpr,label=f'{name} (EER {test_eer:.2%})')
        print(f'{name}: test EER {test_eer:.2%}; AUC {auc:.3f}; FAR {far:.2%}; FRR {frr:.2%}')
    pd.DataFrame(summary).to_csv(OUT / 'final_test_results.csv', index=False)
    test.to_csv(OUT / 'test_trial_scores.csv', index=False)
    plt.plot([0,1],[0,1],'k--',alpha=0.4)
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('True Acceptance Rate (1 - FRR)')
    plt.title('Held-Out Chimeric Verification ROC Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / 'roc_curves.png', dpi=200)
    plt.close()
    print(f'\nFinished. Results folder: {OUT}')
    print('WARNING: Chimeric data do not validate live continuous authentication.')


if __name__ == '__main__':
    main()
