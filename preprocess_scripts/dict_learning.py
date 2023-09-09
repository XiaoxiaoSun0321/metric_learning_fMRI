# %%
import numpy as np
from nilearn import datasets

rest_dataset = datasets.fetch_abide_pcp(data_dir='/mnt/disks/data/')
dsm_field_index = rest_dataset.phenotypic.dtype.names.index('DX_GROUP')
ys = np.stack([p[dsm_field_index] for p in rest_dataset.phenotypic])

site_idx = rest_dataset.phenotypic.dtype.names.index('SITE_ID')
sites = [p[site_idx] for p in rest_dataset.phenotypic]
np.unique(sites, return_counts=True)
#selected_idx = [i for i, p in enumerate(rest_dataset.phenotypic) if p[site_idx].astype(str) == 'NYU']
selected_idx = [i for i, p in enumerate(rest_dataset.phenotypic) if p[site_idx] == 'NYU']

ys = ys[selected_idx]
n_subjects = 150
max_tr = 78
np.unique(ys[:n_subjects], return_counts=True)

# %%
#len(rest_dataset.func_preproc):
fmri_data = np.stack([np.load(f'/mnt/disks/data/abide_processed/abide_data_{i:03}.npy')[:max_tr] for i in range(n_subjects)])
stacked_data = fmri_data.reshape(-1, fmri_data.shape[-1])
print(f'Array size: {fmri_data.size * fmri_data.itemsize * 1e-9:.2f} GB')
print(f'Dict data shape: {stacked_data.shape[0]}x{stacked_data.shape[1]}')

# %%
from sklearn.decomposition import MiniBatchDictionaryLearning
dict_learner = MiniBatchDictionaryLearning(n_components=80, batch_size=12, 
                                           transform_algorithm='lasso_lars', verbose=True)
dict_learner.fit(stacked_data)

# %%
# OLD CODE
# xs = np.stack([dict_learner.transform(seq) for seq in fmri_data])
# ys = ys[:n_subjects]

# %%
# NEW CODE
xs = fmri_data
zs = np.stack([dict_learner.transform(seq) for seq in fmri_data])
ys = ys[:n_subjects]


# %%
import pickle
data = {'zs': zs, 'xs': xs, 'ys':ys, 'dict_learner': dict_learner}
pickle.dump(data, open( "abide_dl_80_data.pkl", "wb" ), protocol=4)



# %%
from sklearn.covariance import EmpiricalCovariance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import classification_report

covs = np.stack([np.tril(EmpiricalCovariance().fit(seq).covariance_, k=0) for seq in xs])
#inv_covs = np.linalg.pinv(covs)
cov_tril = np.stack([cov[np.tril_indices(cov.shape[0], k=0)] for cov in covs])
#cov_tril = np.stack([cov[np.tril_indices(cov.shape[0], k=0)] for cov in inv_covs])

# %%
clf = LogisticRegression(penalty='l2', C=10, solver='liblinear')
cv_results = cross_validate(clf, cov_tril, ys, n_jobs=1, cv=10, return_train_score=True)

vals = []
vals.append(np.mean(cv_results['train_score']))
vals.append(np.std(cv_results['train_score']))
vals.append(np.mean(cv_results['test_score']))
vals.append(np.std(cv_results['test_score']))

print(f'\t{vals[0]:.3f}\u00B1{vals[1]:.3f}' +
      f'\t{vals[2]:.3f}\u00B1{vals[3]:.3f}')



# %%
ys[ys == 1] = 0
ys[ys == 2] = 1

# %%
np.save('abide_150_masked_data.npy', fmri_data)
np.save('abide_150_masked_labels.npy', ys)

# %%
import pickle
data = {'xs': xs, 'ys':ys}
pickle.dump(data, open( "abide_dl_15_data.pkl", "wb" ))
# %%
