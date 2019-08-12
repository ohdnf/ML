#svm_prac.py

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import load_digits
from sklearn import svm, metrics


digits = load_digits()
print(digits.keys())
print(digits['data'][0])
print(digits.images.shape)
print(digits['target_names'])

for i in range(0, 4):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(digits.images[i])
    plt.title('Training: {}'.format(digits.target[i]))
plt.show()

n_samples = len(digits.images)
data_images = digits.images.reshape((n_samples, -1))

x_train_svm, x_test_svm, y_train_svm, y_test_svm = train_test_split(data_images, digits.target)
print('Training data and target sizes: \n{}, {}'.format(x_train_svm.shape, y_train_svm.shape))
print('Test data and target sizes: \n{}, {}'.format(x_test_svm.shape, y_test_svm.shape))

classifier = svm.SVC(gamma=0.001)
# fit to the training data
classifier.fit(x_train_svm, y_train_svm)

y_pred_svm = classifier.predict(x_test_svm)

print("Classification report for classifier %s:\n%s\n" %
     (classifier, metrics.classification_report(y_test_svm, y_pred_svm)))

