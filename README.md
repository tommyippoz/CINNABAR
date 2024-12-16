# CINNABAR
Prediction Rejection Strategies for binary and multi-class classifiers (CINNABAR: prediCtIoN rejectioN strAtegies Binary clAssifieRs)

## Aim/Concept of the Project

Domain experts are desperately looking to solve classification problems by designing and training Machine Learning (ML) models with the highest possible accuracy. 
No matter how hard they try, classifiers will always be prone to misclassifications, especially when dealing with unknown input data. 
This complicates the deployment of classifiers into critical cyber-physical systems (CPSs), where misclassifications could directly impact the health of people, infrastructures, or the environment. 

Here, we consider a classifier as a component of a CPS instead of something to be tested in isolation. 
Such a Critical System Classifier (CSC) may reject outputs it identifies as likely misclassifications, activating system-level mitigation strategies instead. 
Rejections can be triggered by the uncertainty in an input and/or in the prediction of a classifier, which should undergo a cost-sensitive training and being adequately calibrated. 
CSCs achieve superior cost-based performance compared to traditional classifiers, positioning our system-level conceptualization and design of ML classifiers as a crucial and innovative step toward engineering ML classifiers in critical CPSs and infrastructures. 

The proposal is validated by engineering CSCs for hardware failure prediction, error and intrusion detection, which are very common applications in critical CPSs and infrastructures. 
The key findings of this study reveal that: 
(i) a cost-sensitive evaluation produces different outcomes—and leads to different CSC selections—compared to comparisons based on plain classification performance, 
(ii) CSCs demonstrate the potential to effectively handle unknowns, and 
(iii) they are straightforward to deploy using the publicly available library provided in this GitHub.

## Dependencies

CINNABAR needs the following libraries:
- <a href="https://numpy.org/">NumPy</a>
- <a href="https://scipy.org/">SciPy</a>
- <a href="https://pandas.pydata.org/">Pandas</a>
- <a href="https://scikit-learn.org/stable/">Scikit-Learn</a>
- <a href="https://github.com/yzhao062/pyod">PYOD</a>

## Usage

See the 'cinnabar-example.py' file in the 'cinnabar/tests' folder.

The scripts uses data from a study of the same authors (monitoring an IoT device) and shows how to use CINNABAR to execute regular classifiers and to building CSCs with calibration, prediction rejection strategies, and many more.

## Credits

Developed @ University of Trento (Povo), Italy, and University of Florence, Florence, Italy

Contributors
- Tommaso Zoppi
