# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 18:23:38 2017

@author: Work
"""

import numpy as np

from nilearn import datasets
from nilearn import plotting

# By default 2nd subject will be fetched
haxby_dataset = datasets.fetch_haxby()
fmri_filename = haxby_dataset.func[0]

#specific mask from the haxby study
mask_filename = haxby_dataset.mask_vt[0]

#anatomical dataset
anat_filename = haxby_dataset.anat[0]

# Let's visualize it, using the subject's anatomical image as a background
plotting.plot_roi(mask_filename, bg_img=anat_filename, cmap='Paired')

from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=mask_filename, standardize=True)

# We give the masker a filename and retrieve a 2D array ready for machine learning with scikit-learn
fmri_masked = masker.fit_transform(fmri_filename)

#print (fmri_masked)
#print("all condition ",fmri_masked.shape)

# Load target information as string and give a numerical identifier to each
#np.recfromcsv: creates arrays from tabular data
targetInfo_filename = haxby_dataset.session_target[0]
labels = np.recfromcsv(targetInfo_filename, delimiter=" ")

#print(labels)

#Retrieve the behavioral targets, that we are going to predict in the decoding
#???
target = labels['labels']
#print(target[120:130])
#print("all condition ",target.shape)

#create a mask for only the face an cat condition
condition_mask = np.logical_or(target == b'face', target == b'cat')

# We apply this mask in the sampe direction to restrict the classification to the face vs cat discrimination
fmri_masked = fmri_masked[condition_mask]

#print("face&cat condition ",fmri_masked.shape)

#apply the same mask to the target
target = target[condition_mask]
#print(target[100])
#print("face&cat condition ",target.shape)

#create decoder
from sklearn.svm import SVC
svc = SVC(kernel='linear')
#print(svc)

#The svc object is an object that can be fit (or trained) on data with labels, and then predict labels on data without.
#fit on data
svc.fit(fmri_masked, target)

prediction = svc.predict(fmri_masked)
#print(prediction[100])

#measure of error rate
#because we are training on the whole dataset and comparing against the whole data set, it should not have any errors
errRate = (prediction == target).sum() / float(len(target))
#print(errRate)

#cross-validation
svc.fit(fmri_masked[:-30], target[:-30])

prediction = svc.predict(fmri_masked[-30:])
errRate = (prediction == target[-30:]).sum() / float(len(target[-30:]))
#print(errRate)

svc.fit(fmri_masked[::2], target[::2])

prediction = svc.predict(fmri_masked[1::2])
errRate = (prediction == target[::2]).sum() / float(len(target[::2]))
#print(errRate)

#cross-validate repeatedly using KFold
from sklearn.cross_validation import KFold

foldNum = 5
cv = KFold(n=len(fmri_masked), n_folds=foldNum)

for train, test in cv:
    svc.fit(fmri_masked[train], target[train])
    prediction = svc.predict(fmri_masked[test])
    errRate = (prediction == target[test]).sum() / float(len(target[test]))
    #print(errRate)

#use scikit to cross-validate
from sklearn.cross_validation import cross_val_score
cv_score = cross_val_score(svc, fmri_masked, target, cv=cv)
#print(cv_score)

foldMean = np.mean(cv_score)
#print(foldNum,foldMean)

#apply our session mask, to select only cats and faces
session_label = labels['chunks'][condition_mask]

from sklearn.cross_validation import LeaveOneLabelOut
cv = LeaveOneLabelOut(session_label)
cv_score = cross_val_score(svc, fmri_masked, target, cv=cv)
#print(cv_score)
#print(cv_score.size, np.mean(cv_score))

#turn weights into nifti image
coef_ = svc.coef_
#print(coef_)
#print(coef_.shape)

coef_img = masker.inverse_transform(coef_)
#print(coef_img)

coef_img.to_filename('haxby_svc_weights.nii.gz')

#plot the weights against the anatomical background
from nilearn.plotting import plot_stat_map, show

plot_stat_map(coef_img, bg_img=anat_filename,
              title="SVM Weights", display_mode="yx")

show()

